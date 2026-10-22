"""One frozen, paper-separated RoBERTa recipe; final confirmation uses all300claims."""
import argparse
import gc
import math
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.config import load_config
from claim_checker.data.splits import normalize_claim_text
from claim_checker.data.store import (
    append_jsonl,
    load_corpus,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from claim_checker.data.verifier_context import format_premise
from claim_checker.evaluation.analysis import analyse_run
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.runtime import ClaimChecker
from scripts.refit_and_confirm_title_verifier import experiments_for
from scripts.train_verifier import evidence_safe_rows, globally_weighted_losses, training_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/improvement/scifact95_roberta_protocol_v1.json"
SELECT = ROOT / "artifacts/models/roberta_scifact_select_v1"
FINAL = ROOT / "artifacts/models/roberta_scifact_final_v1"
LABELS = {"entailment": "SUPPORT", "contradiction": "CONTRADICT", "neutral": "NEI"}


def audit_rows(cfg, rows, stopping, protocol, final_fit):
    gold = load_gold(cfg.path("processed_dir"), "train")
    corpus = load_corpus(cfg.path("processed_dir"))
    validation = load_gold(cfg.path("processed_dir"), "validation")
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    fit_ids = set(protocol["eligible_ids"] if final_fit else protocol["fit_ids"])
    stop_ids = set() if final_fit else set(protocol["stop_ids"])
    fit_papers = {d for cid in fit_ids for d in gold[cid].reference_doc_ids}
    stop_papers = {d for cid in stop_ids for d in gold[cid].reference_doc_ids}
    assert not fit_ids & stop_ids and not fit_papers & stop_papers
    for population, allowed, key in ((rows, fit_ids, "fit"), (stopping, stop_ids, "stop")):
        assert {r["claim_id"] for r in population} == allowed
        for row in population:
            claim, doc = gold[row["claim_id"]], corpus[row["doc_id"]]
            assert claim.split == "train" and row["claim"] == claim.claim
            assert normalize_claim_text(claim.claim) not in forbidden
            if row["view"] == "hard_negative":
                assert key == "fit" and row["label"] == "NEI" and row["doc_id"] not in claim.reference_doc_ids
            else:
                assert row["doc_id"] in claim.reference_doc_ids
                assert row["label"] == claim.label_by_doc.get(row["doc_id"], "NEI")
            if key == "fit":
                assert row["doc_id"] not in stop_papers
            ids = row.get("context_sentence_ids", list(range(len(doc.sentences))))
            assert row["premise"] == format_premise(doc.title, " ".join(doc.sentences[i] for i in ids), True)
            if row.get("rationale_sentence_ids"):
                assert set(row["rationale_sentence_ids"]) <= set(claim.evidence_by_doc[row["doc_id"]]) & set(ids)
    return {"fit_claims": len(fit_ids), "fit_pairs": len(rows), "stop_claims": len(stop_ids), "stop_pairs": len(stopping),
            "validation_text_overlap": 0, "fit_stop_reference_paper_overlap": 0, "source_annotation_mismatches": 0,
            "weak_negative_pairs": sum(r["view"] == "hard_negative" for r in rows), "final_fit": final_fit}


def train(protocol, *, final_fit=False, pilot=False, physical_batch=2):
    import bitsandbytes as bnb

    cfg, recipe = load_config(), protocol["recipe"]
    assert physical_batch in (1, 2) and recipe["effective_batch"] % physical_batch == 0
    assert protocol["train_gold_sha256"] == sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl")
    assert protocol["validation_gold_sha256"] == sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl")
    base = ROOT / protocol["base_dir"]
    source = read_json(base / "download_manifest.json")
    weights = base / source["weights"]
    frozen_weight = next(f for f in source["files"] if f["name"] == source["weights"])
    assert sha256_file(weights) == frozen_weight["sha256"]
    folder = ROOT / "artifacts/models/roberta_scifact_pilot_v1" if pilot else FINAL if final_fit else SELECT
    selected = read_json(SELECT / "selection.json") if final_fit else None
    if final_fit:
        curve = read_jsonl(SELECT / "learning_curve.jsonl")
        frozen_best = [row for row in curve if row["checkpoint_improved"]][-1]
        assert selected["epoch"] == frozen_best["epoch"] and selected["stop_loss"] == frozen_best["stop_loss"]
        assert selected["validation_used"] is False
        assert selected["weights_sha256"] == sha256_file(SELECT / f"epoch{selected['epoch']}/model.safetensors")
        assert read_json(SELECT / "identity.json")["script_sha256"] == sha256_file(Path(__file__))
    epochs = selected["epoch"] if final_fit else recipe["maximum_epochs"]
    identity = {"protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
                "source_builder_sha256": sha256_file(ROOT / "scripts/train_verifier.py"),
                "context_builder_sha256": sha256_file(ROOT / "src/claim_checker/data/verifier_context.py"),
                "base_weights_sha256": frozen_weight["sha256"], "source_revision": source["revision"],
                "final_fit": final_fit, "epochs": epochs, "physical_batch": physical_batch,
                "selection_sha256": sha256_file(SELECT / "selection.json") if final_fit else None,
                "libraries": {name: version(name) for name in ("torch", "transformers", "bitsandbytes")}}
    if folder.exists():
        assert read_json(folder / "identity.json") == identity, "Changed run: preserve it and create a versioned experiment"
        if (folder / ("pilot.json" if pilot else "status.json")).exists():
            return folder
        raise RuntimeError("Interrupted run: do not mix weights; preserve its artifacts and use a fresh version")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    all_rows, _ = training_rows(cfg, recipe["hard_negatives"], exclude_development=False)
    gold = load_gold(cfg.path("processed_dir"), "train")
    fitting = set(protocol["eligible_ids"] if final_fit else protocol["fit_ids"])
    stop_ids = set() if final_fit else set(protocol["stop_ids"])
    protected = {d for cid in stop_ids for d in gold[cid].reference_doc_ids}
    rows = [r for r in all_rows if r["claim_id"] in fitting and r["doc_id"] not in protected]
    stopping = [r for r in all_rows if r["claim_id"] in stop_ids and r["view"] != "hard_negative"]
    rows, omitted_fit = evidence_safe_rows(cfg, rows, tokenizer, recipe["max_length"], True)
    stopping, omitted_stop = evidence_safe_rows(cfg, stopping, tokenizer, recipe["max_length"], True)
    assert not omitted_fit and not omitted_stop, "Do not train truncated evidence as a decisive label"
    audit = audit_rows(cfg, rows, stopping, protocol, final_fit)
    folder.mkdir(parents=True)
    write_json(folder / "identity.json", identity)
    write_json(folder / "data_audit.json", audit)
    write_jsonl(folder / "training_pairs.jsonl", rows)
    write_jsonl(folder / "stop_pairs.jsonl", stopping)
    print(f"Data audit passed: {audit}", flush=True)
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    assert config.model_type == "roberta"
    config.hidden_dropout_prob = recipe["dropout"]
    config.classifier_dropout = recipe["dropout"]
    model = AutoModelForSequenceClassification.from_pretrained(base, config=config, local_files_only=True,
                                                              weights_only=True, dtype=torch.float32).cuda()
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mapping = {label: config.label2id[name] for name, label in LABELS.items()}

    def encode(values):
        return [(tokenizer(r["premise"], r["claim"], truncation="only_first", max_length=recipe["max_length"]), mapping[r["label"]]) for r in values]

    def collate(batch):
        batch_input = tokenizer.pad([r[0] for r in batch], return_tensors="pt")
        return {k: v.cuda() for k, v in batch_input.items()}, torch.tensor([r[1] for r in batch], device="cuda")

    encoded = encode(rows)
    if pilot:
        encoded.sort(key=lambda row: len(row[0]["input_ids"]), reverse=True)
    loader = DataLoader(encoded, batch_size=physical_batch, shuffle=not pilot, collate_fn=collate)
    stop_loader = DataLoader(encode(stopping), batch_size=physical_batch, collate_fn=collate)
    counts = Counter(r["label"] for r in rows)
    class_weights = torch.empty(3, device="cuda")
    for label, index in mapping.items():
        class_weights[index] = len(rows) / (3 * counts[label])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    accumulation = recipe["effective_batch"] // physical_batch
    horizon = math.ceil(len(loader) / accumulation) * recipe["maximum_epochs"]
    warmup = max(1, horizon // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0.0, (horizon - step) / (horizon - warmup))))
    best_loss, best_epoch, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (batch, labels) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                per_example = globally_weighted_losses(model(**batch).logits, labels, class_weights)
                group_start = step // accumulation * recipe["effective_batch"]
                loss = per_example.sum() / min(recipe["effective_batch"], len(rows) - group_start)
            assert torch.isfinite(loss), "Non-finite training loss"
            loss.backward()
            losses.extend(per_example.detach().float().tolist())
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if pilot and step + 1 >= accumulation:
                write_json(folder / "pilot.json", {"batches": step + 1, "longest_inputs_first": True,
                           "optimizer_step_exercised": True, "finite_loss": True,
                           "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30})
                print("Hardware/gradient pilot passed", flush=True)
                return folder
            if (step + 1) % 100 == 0:
                print(f"{'refit' if final_fit else 'selection'} epoch{epoch}: {step+1}/{len(loader)} batches", flush=True)
        if final_fit:
            append_jsonl(folder / "learning_curve.jsonl", {"epoch": epoch, "weighted_train_loss": float(np.mean(losses))})
            continue
        model.eval()
        stop_losses, actual, predicted = [], [], []
        for batch, labels in stop_loader:
            with torch.inference_mode():
                logits = model(**batch).logits
                stop_losses.extend(torch.nn.functional.cross_entropy(logits, labels, reduction="none").tolist())
                actual.extend(labels.tolist())
                predicted.extend(logits.argmax(-1).tolist())
        inverse = {index: label for label, index in mapping.items()}
        metrics = verdict_metrics([inverse[i] for i in actual], [inverse[i] for i in predicted])
        stop_loss = float(np.mean(stop_losses))
        improved = stop_loss < best_loss - recipe["minimum_loss_improvement"]
        if improved:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            model.save_pretrained(folder / f"epoch{epoch}")
            tokenizer.save_pretrained(folder / f"epoch{epoch}")
        else:
            stale += 1
        report = {"epoch": epoch, "weighted_train_loss": float(np.mean(losses)), "stop_loss": stop_loss,
                  "stop_pair_accuracy": metrics["accuracy"], "stop_pair_macro_f1": metrics["macro_f1"],
                  "checkpoint_improved": improved, "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "note": "Reference-supplied TRAIN stopping component; not validation300 accuracy"}
        append_jsonl(folder / "learning_curve.jsonl", report)
        print(report, flush=True)
        if stale >= recipe["patience"]:
            break
    if final_fit:
        best_epoch = epochs
        model.save_pretrained(folder / f"epoch{epochs}")
        tokenizer.save_pretrained(folder / f"epoch{epochs}")
    write_json(folder / "selection.json", {"epoch": best_epoch, "stop_loss": None if final_fit else best_loss,
               "validation_used": False, "weights_sha256": sha256_file(folder / f"epoch{best_epoch}/model.safetensors")})
    write_json(folder / "status.json", {"complete": True, "final_fit": final_fit, "selected_epoch": best_epoch})
    del model, optimizer, trainable, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return folder


def confirm(protocol):
    selection = read_json(FINAL / "selection.json")
    model_dir = FINAL / f"epoch{selection['epoch']}"
    assert selection["weights_sha256"] == sha256_file(model_dir / "model.safetensors")
    settings = protocol["inference"]
    profile = {"product": {"judge_backend": "nli", "evidence_mode": "full_abstract", "judge_think": False},
               "retrieval": {"pipeline_k": settings["k"]},
               "verifier": {"model_dir": str(model_dir.relative_to(ROOT)), "device": "cpu", "include_title": True,
                            "window_overlap": settings["window_overlap"], "min_retrieval_score": settings["gate"],
                            "support_threshold": settings["support_threshold"], "contradict_threshold": settings["contradict_threshold"]}}
    path = ROOT / "configs/verifier_roberta_candidate.yaml"
    if path.exists():
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == profile
    else:
        path.write_text("# Experimental fixed-recipe RoBERTa reader; not a claimed95% result.\n" + yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    cfg = load_config(path, overrides={"verifier": {"device": "cuda"}})
    experiments = experiments_for(settings["k"])
    checker = ClaimChecker(cfg, experiments=experiments)
    records, subsets = [], []
    for cohort in ("holdout", "recheck", "oracle"):
        run_id = f"roberta_scifact_{cohort}_v1"
        folder = run_evaluation(cfg, experiments, checker, suite=f"title_{cohort}", run_id=run_id)
        analyse_run(folder, checker.corpus)
        rows = read_jsonl(folder / "predictions.jsonl")
        metrics = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
        subsets.append({"run_id": run_id, "metrics": metrics, "operational_errors": sum(r["status"] != "ok" for r in rows)})
        if cohort != "oracle":
            records.extend(rows)
        print(f"Completed {run_id}: {metrics['accuracy']:.4%} / {metrics['macro_f1']:.6f}", flush=True)
    assert len(records) == len({r["claim_id"] for r in records}) == 300
    pooled = verdict_metrics([r["gold_label"] for r in records], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in records])
    report = {"protocol_sha256": sha256_file(PROTOCOL), "subsets": subsets, "pooled_validation_300": pooled,
              "population_note": "Same reused validation300, not a new blind test; separate oracle excluded from pooled metrics",
              "target_met": pooled["accuracy"] > 0.95 and pooled["macro_f1"] > 0.95,
              "baseline_correct": 242, "candidate_correct": pooled["correct"]}
    write_json(ROOT / "artifacts/improvement/final_roberta_verifier_results.json", report)
    print(f"Actual validation300: {pooled['accuracy']:.4%} / {pooled['macro_f1']:.6f}; target_met={report['target_met']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "select", "refit", "confirm", "all"), required=True)
    parser.add_argument("--physical-batch", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = read_json(PROTOCOL)
    if args.stage == "pilot":
        train(protocol, pilot=True, physical_batch=args.physical_batch)
    if args.stage in ("select", "all"):
        train(protocol, physical_batch=args.physical_batch)
    if args.stage in ("refit", "all"):
        train(protocol, final_fit=True, physical_batch=args.physical_batch)
    if args.stage in ("confirm", "all"):
        confirm(protocol)


if __name__ == "__main__":
    main()
