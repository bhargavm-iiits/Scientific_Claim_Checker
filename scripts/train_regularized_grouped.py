"""One fixed, regularized NLI recipe with isolated TRAIN stop/audit paper groups.

Historical validation scores never enter this trainer. The audit predictions are
made only after checkpoint selection is frozen. They are not the 300-claim target.
"""
import argparse
import gc
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.config import load_config
from claim_checker.data.store import (
    append_jsonl,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from claim_checker.evaluation.analysis import variant_metrics
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.evaluation.runner import prediction_record
from claim_checker.runtime import ClaimChecker
from scripts.train_verifier import evidence_safe_rows, globally_weighted_losses, training_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"


def fit_fold(protocol, split, all_rows):
    import bitsandbytes as bnb

    recipe = protocol["recipe"]
    cfg = load_config()
    gold = load_gold(cfg.path("processed_dir"), "train")
    protected = {d for cid in split["stop"] + split["audit"] for d in gold[cid].reference_doc_ids}
    selected = [r for r in all_rows if r["claim_id"] in set(split["train"]) and r["doc_id"] not in protected]
    stopping = [r for r in all_rows if r["claim_id"] in set(split["stop"]) and r["view"] != "hard_negative"]
    assert not set(split["train"]) & (set(split["stop"]) | set(split["audit"]))
    folder = ROOT / f"artifacts/models/regularized_grouped_v1_fold{split['fold']}"
    identity = {"protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
                "source_builder_sha256": sha256_file(ROOT / "scripts/train_verifier.py"),
                "base_weights_sha256": sha256_file(ROOT / recipe["base_dir"] / "model.safetensors"),
                "fold": split["fold"]}
    if folder.exists():
        assert read_json(folder / "identity.json") == identity, "Changed experiment; refuse resume"
        if not (folder / "selection.json").exists():
            raise RuntimeError("Training was interrupted; preserve artifacts and use a new named protocol")
        return folder
    folder.mkdir(parents=True)
    write_json(folder / "identity.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / recipe["base_dir"], local_files_only=True)
    selected, train_omitted = evidence_safe_rows(cfg, selected, tokenizer, 512, True)
    stopping, stop_omitted = evidence_safe_rows(cfg, stopping, tokenizer, 512, True)
    assert not train_omitted and not stop_omitted, "Do not silently omit training or stopping cases"
    write_jsonl(folder / "training_pairs.jsonl", selected)
    write_jsonl(folder / "stop_pairs.jsonl", stopping)
    write_json(folder / "integrity.json", {
        "training_claims": len({r["claim_id"] for r in selected}), "training_pairs": len(selected),
        "stopping_claims": len({r["claim_id"] for r in stopping}), "stopping_pairs": len(stopping),
        "audit_claims": len(split["audit"]), "reference_paper_overlap": 0, "normalized_text_overlap": 0,
        "training_pairs_sha256": sha256_file(folder / "training_pairs.jsonl"),
        "stop_pairs_sha256": sha256_file(folder / "stop_pairs.jsonl"),
        "weak_negative_pairs": sum(r["view"] == "hard_negative" for r in selected),
        "note": "BM25 nonreference negatives are weak labels; no held-out stop/audit papers enter these negatives.",
    })
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(ROOT / recipe["base_dir"], local_files_only=True)
    config.hidden_dropout_prob = recipe["dropout"]
    config.pooler_dropout = recipe["dropout"]
    config.cls_dropout = recipe["dropout"]
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / recipe["base_dir"], config=config,
                                                              local_files_only=True, dtype=torch.float32).cuda()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in [*list(model.deberta.encoder.layer)[-recipe["layers"]:], model.pooler, model.classifier]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mapping = {"SUPPORT": config.label2id["entailment"], "CONTRADICT": config.label2id["contradiction"], "NEI": config.label2id["neutral"]}

    def encode(rows):
        result = []
        for row in rows:
            tokens = tokenizer(row["premise"], row["claim"], truncation="only_first", max_length=512)
            result.append((tokens, mapping[row["label"]]))
        return result

    def collate(batch):
        inputs = tokenizer.pad([r[0] for r in batch], return_tensors="pt")
        return {k: v.cuda() for k, v in inputs.items()}, torch.tensor([r[1] for r in batch], device="cuda")

    loader = DataLoader(encode(selected), batch_size=recipe["physical_batch"], shuffle=True, collate_fn=collate)
    stop_loader = DataLoader(encode(stopping), batch_size=recipe["physical_batch"], collate_fn=collate)
    counts = Counter(r["label"] for r in selected)
    weights = torch.empty(3, device="cuda")
    for label, index in mapping.items():
        weights[index] = len(selected) / (3 * counts[label])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    accumulation = recipe["effective_batch"] // recipe["physical_batch"]
    horizon = math.ceil(len(loader) / accumulation) * recipe["maximum_epochs"]
    warmup = max(1, horizon // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0.0, (horizon - step) / (horizon - warmup))))
    best_loss, best_epoch, stale = float("inf"), None, 0
    for epoch in range(1, recipe["maximum_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (batch, labels) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                per_example = globally_weighted_losses(model(**batch).logits, labels, weights)
                group_start = step // accumulation * recipe["effective_batch"]
                loss = per_example.sum() / min(recipe["effective_batch"], len(selected) - group_start)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            losses.extend(per_example.detach().float().tolist())
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        stop_losses, actual, predicted = [], [], []
        for batch, labels in stop_loader:
            with torch.inference_mode():
                logits = model(**batch).logits
                stop_losses.extend(torch.nn.functional.cross_entropy(logits, labels, reduction="none").tolist())
                actual.extend(labels.tolist())
                predicted.extend(logits.argmax(-1).tolist())
        stop_loss = float(np.mean(stop_losses))
        inverse = {index: label for label, index in mapping.items()}
        metrics = verdict_metrics([inverse[i] for i in actual], [inverse[i] for i in predicted])
        report = {"epoch": epoch, "weighted_train_loss": float(np.mean(losses)), "unweighted_stop_loss": stop_loss,
                  "stop_pair_accuracy": metrics["accuracy"], "stop_pair_macro_f1": metrics["macro_f1"],
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "note": "Reference-supplied component metrics, not application claim accuracy"}
        improved = stop_loss < best_loss - recipe["minimum_loss_improvement"]
        if improved:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            model.save_pretrained(folder / f"epoch{epoch}")
            tokenizer.save_pretrained(folder / f"epoch{epoch}")
        else:
            stale += 1
        report["checkpoint_improved"] = improved
        append_jsonl(folder / "learning_curve.jsonl", report)
        print(f"fold{split['fold']} {report}", flush=True)
        if stale >= recipe["patience"]:
            break
    write_json(folder / "selection.json", {"epoch": best_epoch, "stop_loss": best_loss,
               "selection_rule": recipe["selection"], "audit_used": False,
               "weights_sha256": sha256_file(folder / f"epoch{best_epoch}/model.safetensors")})
    del model, optimizer, trainable, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return folder


def audit_fold(protocol, split, folder):
    selected = read_json(folder / "selection.json")
    model = folder / f"epoch{selected['epoch']}"
    assert selected["weights_sha256"] == sha256_file(model / "model.safetensors")
    recipe = protocol["recipe"]
    cfg = load_config(overrides={"product": {"judge_backend": "nli", "evidence_mode": "full_abstract"},
                                "verifier": {"model_dir": str(model.relative_to(ROOT)), "device": "cuda", "include_title": True,
                                             "window_overlap": 2, "support_threshold": recipe["support_threshold"],
                                             "contradict_threshold": recipe["contradict_threshold"], "min_retrieval_score": recipe["retrieval_gate"]},
                                "retrieval": {"pipeline_k": recipe["k"]}})
    checker = ClaimChecker(cfg)
    train = load_gold(cfg.path("processed_dir"), "train")
    path = folder / "audit_predictions.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in existing}
    assert done <= set(split["audit"]) and len(done) == len(existing)
    for index, cid in enumerate(split["audit"], 1):
        if cid in done:
            continue
        result = checker.check(train[cid].claim, claim_id=cid)
        append_jsonl(path, prediction_record("regularized_grouped", train[cid], result))
        print(f"fold{split['fold']} actual application audit {index}/{len(split['audit'])}; status={result.status}", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == len(split["audit"])
    metrics = variant_metrics(rows, checker.corpus, None)
    write_json(folder / "audit_metrics.json", {"population": "Held-out original TRAIN paper groups; not external validation300",
               "selection_sha256": sha256_file(folder / "selection.json"), "metrics": metrics})
    del checker
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="Protocol fold indices; default first-fold pilot")
    args = parser.parse_args()
    protocol = read_json(PROTOCOL)
    if any(fold not in range(protocol["n_folds"]) for fold in args.folds):
        parser.error("Unknown protocol fold")
    torch.set_num_threads(4)
    all_rows, _ = training_rows(load_config(), protocol["recipe"]["hard_negatives"], exclude_development=False)
    for number in args.folds:
        split = protocol["folds"][number]
        folder = fit_fold(protocol, split, all_rows)
        audit_fold(protocol, split, folder)
    print("Completed requested grouped TRAIN audits. These are not the requested external-validation scores.", flush=True)


if __name__ == "__main__":
    main()
