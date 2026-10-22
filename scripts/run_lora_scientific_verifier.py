"""One frozen all-layer LoRA experiment with hard negatives and evidence rescue."""
import argparse
import gc
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.config import load_config, sha256_text
from claim_checker.data.splits import load_split, normalize_claim_text
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
from claim_checker.evaluation.analysis import variant_metrics
from claim_checker.evaluation.metrics import evidence_metrics
from claim_checker.evaluation.runner import prediction_record
from claim_checker.evaluation.training_provenance import assert_unfitted_population
from claim_checker.retrieval.index import check_index
from claim_checker.runtime import ClaimChecker
from scripts.lora_scientific_verifier import (
    FeedbackChecker,
    adapt_encoder,
    row_weight,
    weak_sentence_negatives,
)
from scripts.train_verifier import evidence_safe_rows, training_rows

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/models/nli_large_base"
PLAN = ROOT / "artifacts/improvement/lora_science_protocol_v1.json"
SELECT = ROOT / "artifacts/models/lora_science_select_v1"
FINAL = ROOT / "artifacts/models/lora_science_final_v1"


def experiment_config(model_dir=None):
    overrides = {"retrieval": {"candidates_per_method": 100, "rerank_top_n": 50, "keep_top": 20}}
    if model_dir is not None:
        overrides.update(product={"judge_backend": "nli", "evidence_mode": "full_abstract"},
                         verifier={"model_dir": str(model_dir), "device": "cuda"})
    return load_config(overrides=overrides)


def prepare():
    parent = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    grouped = read_json(parent)
    split = grouped["folds"][0]
    cfg = experiment_config()
    train, validation = load_gold(cfg.path("processed_dir"), "train"), load_gold(cfg.path("processed_dir"), "validation")
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    assert all(normalize_claim_text(train[c].claim) not in forbidden for c in grouped["group_by_claim"])
    baseline = read_json(ROOT / "artifacts/improvement/polarity_audit_v1/paired_quality.json")
    source_paths = [Path(__file__), ROOT / "scripts/lora_scientific_verifier.py", ROOT / "scripts/train_verifier.py"]
    source_paths += sorted((ROOT / "src/claim_checker").rglob("*.py"))
    plan = {"name": "lora_science_v1", "goal": "All full-pipeline quality metrics >0.95 on original validation300; oracle100 remains separate",
            "fit_ids": split["train"], "stop_ids": split["stop"], "audit_ids": split["audit"],
            "eligible_ids": sorted(grouped["group_by_claim"]), "group_protocol_sha256": sha256_file(parent),
            "base_revision": (BASE / "source_revision.txt").read_text(), "base_weights_sha256": sha256_file(BASE / "model.safetensors"),
            "base_config_sha256": sha256_file(BASE / "config.json"), "base_tokenizer_sha256": sha256_file(BASE / "tokenizer.json"),
            "retrieval_config": cfg.retrieval.model_dump(mode="json"), "index_fingerprint": check_index(cfg)["fingerprint"],
            "default_config_sha256": sha256_file(ROOT / "configs/default.yaml"),
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
            "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
            "raw_train_sha256": sha256_file(cfg.path("raw_dir") / "scifact/claims_train.jsonl"),
            "corpus_sha256": sha256_file(cfg.path("processed_dir") / "corpus.jsonl"),
            "stop_reference_pairs_sha256": sha256_file(ROOT / "artifacts/models/regularized_grouped_v1_fold0/stop_pairs.jsonl"),
            "pilot_fit_reference_pairs_sha256": sha256_file(ROOT / "artifacts/models/regularized_grouped_v1_fold0/training_pairs.jsonl"),
            "validation_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
            "recipe": {"seed": 42, "rank": 16, "alpha": 32, "lora_dropout": .1, "backbone_dropout": .2,
                       "target_modules": ["query_proj", "value_proj"], "all_encoder_layers": True,
                       "heads": ["pooler", "classifier"], "lr": 5e-5, "weight_decay": .05,
                       "max_epochs": 4, "patience": 1, "min_delta": .002, "physical_batch": 2, "effective_batch": 16,
                       "max_length": 512, "class_weight": "sqrt inverse-frequency on gradient-fit rows only",
                       "nonreference_weight": .25, "nonrationale_weight": .1, "optimizer": "AdamW8bit",
                       "precision": "FP32 frozen weights/adapters; BF16 CUDA autocast; non-reentrant gradient checkpointing"},
            "data": "Reference abstract/rationale pairs; one genuine widened hybrid+reranked top20 hard nonreference per TRAIN fitting claim excluding stop/audit papers; one lexical unannotated sentence per positive fitting claim/paper, weak NEI",
            "selection": "Mean unweighted CE on160 stopping claims'292 reference pairs, min_delta0.002 patience1; audit never chooses epoch",
            "inference": {"k": 3, "gate": -2, "support_cutoff": .5, "contradict_cutoff": .9,
                          "rescue": "Only if initial top3 yield no decisive label or window conflict, inspect remaining genuine top20 without rerank gate; retain best source per direction >=0.9, including opposing evidence/window conflicts, within3 reporting slots; all scored sources logged separately",
                          "rescue_cutoff": .9, "window_overlap": 2, "include_title": True},
            "baseline": {"verdict": baseline["metrics"]["control"]["verdict"], "quality": baseline["quality"]["control"]},
            "audit_gate": "Both accuracy and MacroF1 >=saved baseline+0.02; citation P/R and retrieval R3/R10 do not decline; zero errors; all quotes exact",
            "refit": "Only on audit pass: fresh generic base, same LoRA recipe on807eligibleTRAIN for selectedepochs, original4epochLRhorizon",
            "confirmation": "Unchanged original300 retrieved, separate100oracle; no subset changes or threshold grid",
            "new_training_recipes": 1, "new_threshold_searches": 0,
            "references": ["https://arxiv.org/abs/2106.09685", "https://arxiv.org/abs/2508.11122"],
            "limitations": "A new experimental NLI backend, not the original Qwen judge. Reused TRAIN audit and validation, not blind. Weak negatives can contain unannotated evidence. All-layer LoRA and data changes are a combined candidate, not a causal ablation."}
    if PLAN.exists():
        assert read_json(PLAN) == plan, "Frozen protocol changed; retain old results and use a new version"
    else:
        write_json(PLAN, plan)
    return plan


def data(plan, final=False):
    cfg = experiment_config()
    corpus, gold = load_corpus(cfg.path("processed_dir")), load_gold(cfg.path("processed_dir"), "train")
    folder = ROOT / f"artifacts/improvement/lora_science_{'final' if final else 'selection'}_data_v1"
    folder.mkdir(exist_ok=True)
    identity = {"plan_sha256": sha256_file(PLAN), "final": final, "index": check_index(cfg)["fingerprint"]}
    if (folder / "identity.json").exists():
        assert read_json(folder / "identity.json") == identity
    else:
        write_json(folder / "identity.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    fit_ids = set(plan["eligible_ids"] if final else plan["fit_ids"])
    protected = set() if final else {d for cid in plan["stop_ids"] + plan["audit_ids"] for d in gold[cid].reference_doc_ids}
    rows, _ = training_rows(cfg, 0, exclude_development=False)
    rows = [r for r in rows if r["claim_id"] in fit_ids]
    assert all(r["doc_id"] not in protected for r in rows)
    rows, omitted = evidence_safe_rows(cfg, rows, tokenizer, 512, True)
    assert not omitted
    hits_path = folder / "fit_retrieval_hits.jsonl"
    mined = read_jsonl(hits_path) if hits_path.exists() else []
    done = {r["claim_id"] for r in mined}
    assert len(done) == len(mined) and done <= fit_ids
    if done != fit_ids:
        checker = ClaimChecker(cfg)
        for index, cid in enumerate(sorted(fit_ids), 1):
            if cid not in done:
                hits = checker._retrieve(gold[cid].claim, "hybrid", True, 20).hits
                candidates = [h for h in hits if h.doc_id not in set(gold[cid].reference_doc_ids) | protected]
                append_jsonl(hits_path, {"claim_id": cid, "hits": [h.model_dump(mode="json") for h in hits],
                                       "negative_doc_id": candidates[0].doc_id if candidates else None})
            if index % 25 == 0:
                print(f"TRAIN negative mining: {index}/{len(fit_ids)}", flush=True)
        del checker
        gc.collect()
        mined = read_jsonl(hits_path)
    weak = weak_sentence_negatives(rows, corpus, gold)
    for row in mined:
        assert len({h["doc_id"] for h in row["hits"]}) == len(row["hits"]) <= 20
        if row["negative_doc_id"] is None:
            continue
        cid, did = row["claim_id"], row["negative_doc_id"]
        assert did in {h["doc_id"] for h in row["hits"]}, "Training negative must have been genuinely retrieved"
        assert did not in set(gold[cid].reference_doc_ids) | protected
        doc = corpus[did]
        rows.append({"claim_id": cid, "claim": gold[cid].claim, "doc_id": did,
                     "premise": format_premise(doc.title, " ".join(doc.sentences), True), "label": "NEI",
                     "view": "hard_negative", "context_sentence_ids": list(range(len(doc.sentences)))})
    rows.extend(weak)
    stopping = [] if final else read_jsonl(ROOT / "artifacts/models/regularized_grouped_v1_fold0/stop_pairs.jsonl")
    assert {r["claim_id"] for r in rows} == fit_ids
    if not final:
        assert {r["claim_id"] for r in stopping} == set(plan["stop_ids"])
        assert not {r["doc_id"] for r in rows} & {r["doc_id"] for r in stopping}
    for role, population in (("fit", rows), ("stop", stopping)):
        for row in population:
            g, doc = gold[row["claim_id"]], corpus[row["doc_id"]]
            assert g.split == "train" and row["claim"] == g.claim
            ids = row.get("context_sentence_ids", list(range(len(doc.sentences))))
            assert ids and ids == sorted(set(ids)) and all(0 <= i < len(doc.sentences) for i in ids)
            assert row["premise"] == format_premise(doc.title, " ".join(doc.sentences[i] for i in ids), True)
            if row["view"] == "hard_negative":
                assert doc.doc_id not in g.reference_doc_ids and row["label"] == "NEI"
            elif row["view"] == "weak_nonrationale":
                assert not set(ids) & set(g.evidence_by_doc.get(doc.doc_id, [])) and row["label"] == "NEI"
            else:
                assert doc.doc_id in g.reference_doc_ids and row["label"] == g.label_by_doc.get(doc.doc_id, "NEI")
            if row["label"] != "NEI":
                assert len(tokenizer(row["premise"], row["claim"])["input_ids"]) <= 512
        write_jsonl(folder / f"{role}_rows.jsonl", population)
    integrity = {"identity": identity, "fit_claims": len(fit_ids), "fit_rows": len(rows), "stop_rows": len(stopping),
                 "views": dict(Counter(r["view"] for r in rows)), "labels": dict(Counter(r["label"] for r in rows)),
                 "source_reconstruction_mismatches": 0, "positive_truncations": 0,
                 "fit_stop_audit_source_paper_overlap": None if final else 0,
                 "fit_rows_sha256": sha256_file(folder / "fit_rows.jsonl"), "stop_rows_sha256": sha256_file(folder / "stop_rows.jsonl")}
    integrity["retrieval_hits_sha256"] = sha256_file(hits_path)
    write_json(folder / "integrity.json", integrity)
    print(f"Data verified: {integrity}", flush=True)
    return tokenizer, rows, stopping


def new_model(recipe):
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    config.hidden_dropout_prob = config.pooler_dropout = config.cls_dropout = recipe["backbone_dropout"]
    base = AutoModelForSequenceClassification.from_pretrained(BASE, config=config, local_files_only=True, dtype=torch.float32)
    model = adapt_encoder(base, rank=recipe["rank"], alpha=recipe["alpha"], dropout=recipe["lora_dropout"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.cuda()


def fit(plan, final=False, pilot=False):
    import bitsandbytes as bnb

    recipe = plan["recipe"]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if pilot:
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
        rows = read_jsonl(ROOT / "artifacts/models/regularized_grouped_v1_fold0/training_pairs.jsonl")
        assert {r["claim_id"] for r in rows} == set(plan["fit_ids"])
        stop = []
    else:
        tokenizer, rows, stop = data(plan, final)
    folder = FINAL if final else SELECT
    identity = {"plan_sha256": sha256_file(PLAN), "final": final}
    if not pilot:
        if folder.exists():
            assert read_json(folder / "identity.json") == identity
            assert (folder / "selection.json").exists(), "Interrupted fitting: preserve it and use a new named experiment"
            return
        folder.mkdir(parents=True)
        write_json(folder / "identity.json", identity)
        gold = load_gold(load_config().path("processed_dir"), "train")
        used = sorted({r["claim_id"] for r in rows + stop})
        write_json(folder / "training_manifest.json", {"fitted_claim_ids": used,
                   "fitted_claim_text_sha256": sorted({sha256_text(normalize_claim_text(gold[c].claim)) for c in used}),
                   "gradient_fit_claim_ids": sorted({r["claim_id"] for r in rows}),
                   "stopping_claim_ids": sorted({r["claim_id"] for r in stop}), "plan_sha256": sha256_file(PLAN),
                   "source_split": "train", "final_fit": final})
    model = new_model(recipe)
    mapping = {"SUPPORT": 0, "NEI": 1, "CONTRADICT": 2}
    assert model.config.label2id == {"entailment": 0, "neutral": 1, "contradiction": 2}

    def encode(population):
        return [{"tokens": tokenizer(r["premise"], r["claim"], truncation="only_first", max_length=512),
                 "label": mapping[r["label"]], "weight": row_weight(r)} for r in population]

    def collate(population):
        batch = tokenizer.pad([r["tokens"] for r in population], return_tensors="pt")
        return ({k: v.cuda() for k, v in batch.items()}, torch.tensor([r["label"] for r in population], device="cuda"),
                torch.tensor([r["weight"] for r in population], device="cuda"))

    encoded, stopping = encode(rows), encode(stop)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    if pilot:
        longest = sorted(encoded, key=lambda r: len(r["tokens"]["input_ids"]), reverse=True)[:2]
        inputs, labels, _ = collate(longest)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(**inputs).logits.float(), labels)
        assert torch.isfinite(loss)
        loss.backward()
        nonzero_layers = {int(m.group(1)) for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0
                          and (m := re.search(r"encoder\.layer\.(\d+)\..*\.lora_B\.", name))}
        assert nonzero_layers == set(range(24)), nonzero_layers
        optimizer.step()
        write_json(ROOT / "artifacts/improvement/lora_science_hardware_v1.json", {
            "longest_tokens": [len(r["tokens"]["input_ids"]) for r in longest], "finite_loss": True,
            "nonzero_gradient_encoder_layers": sorted(nonzero_layers), "trainable_parameters": sum(p.numel() for p in trainable),
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "checkpoint_saved": False})
        print("All24-layer LoRA hardware/backprop pilot passed; no checkpoint saved.", flush=True)
        return
    counts = Counter(r["label"] for r in rows)
    balance = torch.tensor([math.sqrt(len(rows) / (3 * counts[label])) for label in ("SUPPORT", "NEI", "CONTRADICT")], device="cuda")
    loader = torch.utils.data.DataLoader(encoded, batch_size=2, shuffle=True, collate_fn=lambda batch: batch)
    accumulation = 8
    horizon = math.ceil(len(loader) / accumulation) * recipe["max_epochs"]
    warmup = max(1, horizon // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0., (horizon - step) / (horizon - warmup))))
    epochs = read_json(SELECT / "selection.json")["epoch"] if final else recipe["max_epochs"]
    best, best_epoch, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, current in enumerate(loader):
            batch, labels, weak = collate(current)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = torch.nn.functional.cross_entropy(model(**batch).logits.float(), labels, reduction="none") * balance[labels] * weak
                loss = values.sum() / min(16, len(encoded) - step // accumulation * 16)
            assert torch.isfinite(loss)
            loss.backward()
            losses.extend(values.detach().tolist())
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 150 == 0:
                print(f"LoRA {'refit' if final else 'fit'} epoch{epoch}: {step + 1}/{len(loader)} batches", flush=True)
        model.eval()
        stop_losses = []
        for start in range(0, len(stopping), 2):
            batch, labels, _ = collate(stopping[start:start + 2])
            with torch.inference_mode():
                stop_losses.extend(torch.nn.functional.cross_entropy(model(**batch).logits.float(), labels, reduction="none").tolist())
        stop_loss = float(np.mean(stop_losses)) if stopping else None
        assert stop_loss is None or math.isfinite(stop_loss)
        improved = final or stop_loss < best - recipe["min_delta"]
        if improved:
            best, best_epoch, stale = stop_loss, epoch, 0
            if not final or epoch == epochs:
                model.save_pretrained(folder / f"adapter_epoch{epoch}")
        else:
            stale += 1
        report = {"epoch": epoch, "weighted_train_loss": float(np.mean(losses)), "unweighted_stop_ce": stop_loss,
                  "improved": improved, "elapsed_seconds": time.perf_counter() - started,
                  "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
        append_jsonl(folder / "learning_curve.jsonl", report)
        print(report, flush=True)
        if not final and stale >= recipe["patience"]:
            break
    write_json(folder / "selection.json", {"epoch": best_epoch, "stop_loss": best, "validation_used": False,
               "adapter_sha256": sha256_file(folder / f"adapter_epoch{best_epoch}/adapter_model.safetensors")})


def export(plan, final=False):
    folder = FINAL if final else SELECT
    selection = read_json(folder / "selection.json")
    if (folder / "merged/model.safetensors").exists():
        assert sha256_file(folder / "merged/model.safetensors") == selection["merged_weights_sha256"]
        return
    assert sha256_file(folder / f"adapter_epoch{selection['epoch']}/adapter_model.safetensors") == selection["adapter_sha256"]
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    config.hidden_dropout_prob = config.pooler_dropout = config.cls_dropout = plan["recipe"]["backbone_dropout"]
    base = AutoModelForSequenceClassification.from_pretrained(BASE, config=config, local_files_only=True, dtype=torch.float32)
    adapted = PeftModel.from_pretrained(base, folder / f"adapter_epoch{selection['epoch']}", local_files_only=True)
    merged = adapted.merge_and_unload(safe_merge=True).eval()
    merged.save_pretrained(folder / "merged")
    AutoTokenizer.from_pretrained(BASE, local_files_only=True).save_pretrained(folder / "merged")
    selection["merged_weights_sha256"] = sha256_file(folder / "merged/model.safetensors")
    write_json(folder / "selection.json", selection)


def quality_target(metrics):
    scores = {"accuracy": metrics["verdict"]["accuracy"], "macro_f1": metrics["verdict"]["macro_f1"]}
    for label, values in metrics["verdict"]["per_class"].items():
        scores.update({f"{label}_{name}": values[name] for name in ("precision", "recall", "f1")})
    scores.update({f"evidence_{name}": metrics["evidence_cited"][name] for name in ("precision", "recall")})
    scores.update({key: metrics["retrieval"][key] for key in ("recall@3", "recall@10")} if "retrieval" in metrics else {})
    scores["quote_validity"] = metrics["quotes"]["validity"]
    return {"scores": scores, "all_above_0_95": all(value is not None and value > .95 for value in scores.values()),
            "primary_accuracy_and_macro_above_0_95": scores["accuracy"] > .95 and scores["macro_f1"] > .95}


def evaluate(plan, final=False, oracle=False):
    folder = FINAL if final else SELECT
    selection = read_json(folder / "selection.json")
    assert sha256_file(folder / "merged/model.safetensors") == selection["merged_weights_sha256"]
    cfg = experiment_config(folder / "merged")
    checker = FeedbackChecker(cfg, folder / "merged")
    gold = load_gold(cfg.path("processed_dir"), "validation" if final else "train")
    ids = (load_split(cfg, "final")["claim_ids"] if oracle else sorted(gold)) if final else plan["audit_ids"]
    assert_unfitted_population(read_json(folder / "training_manifest.json"), [gold[c] for c in ids])
    stage = "oracle100" if oracle else "validation300" if final else "audit163"
    destination = ROOT / f"artifacts/improvement/lora_science_{stage}_v1"
    destination.mkdir(exist_ok=True)
    identity = {"plan_sha256": sha256_file(PLAN), "selection": selection, "ids": ids,
                "index": None if oracle else check_index(cfg)["fingerprint"]}
    if (destination / "identity.json").exists():
        assert read_json(destination / "identity.json") == identity
    else:
        write_json(destination / "identity.json", identity)
    path = destination / "predictions.jsonl"
    saved = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in saved}
    assert len(done) == len(saved) and done <= set(ids)
    for index, cid in enumerate(ids, 1):
        if cid not in done:
            g = gold[cid]
            result = checker.check(g.claim, cid, g.reference_doc_ids if oracle else None)
            record = prediction_record("lora_science_v1", g, result)
            record["reported_doc_ids"] = record["examined_doc_ids"]
            record["examined_doc_ids"] = result.metadata.get("scored_candidate_doc_ids", [])
            record["feedback_diagnostics"] = {key: result.metadata.get(key) for key in
                                               ("initial_retrieval_hits", "rescued_doc_ids", "rescue_attempted", "candidate_judgements")}
            append_jsonl(path, record)
        if index % 20 == 0:
            print(f"LoRA {stage}: {index}/{len(ids)}", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == len(ids) and {r["claim_id"] for r in rows} == set(ids)
    metrics = variant_metrics(rows, checker.corpus, None)
    metrics.setdefault("evidence_cited", evidence_metrics(rows, "cited"))
    report = {"population": stage, "reused_not_blind": True, "metrics": metrics,
              "target": quality_target(metrics), "rescued_claims": sum(bool(r["feedback_diagnostics"].get("rescued_doc_ids")) for r in rows)}
    if not final:
        base = plan["baseline"]
        report["gate_conditions"] = {
            "accuracy_gain": metrics["verdict"]["accuracy"] >= base["verdict"]["accuracy"] + .02,
            "macro_gain": metrics["verdict"]["macro_f1"] >= base["verdict"]["macro_f1"] + .02,
            "citation_precision": (metrics["evidence_cited"]["precision"] or 0.) >= base["quality"]["citations"]["precision"],
            "citation_recall": (metrics["evidence_cited"]["recall"] or 0.) >= base["quality"]["citations"]["recall"],
            "retrieval3": metrics["retrieval"]["recall@3"] >= base["quality"]["retrieval"]["recall@3"],
            "retrieval10": metrics["retrieval"]["recall@10"] >= base["quality"]["retrieval"]["recall@10"],
            "zero_errors": metrics["operational"]["errors"] == 0, "exact_quotes": metrics["quotes"]["validity"] == 1.,
        }
        report["audit_gate_passed"] = all(report["gate_conditions"].values())
    write_json(destination / "metrics.json", report)
    print(f"LoRA {stage}: {metrics['verdict']['accuracy']:.4%}/{metrics['verdict']['macro_f1']:.6f}, errors={metrics['operational']['errors']}", flush=True)
    return report


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "data", "pilot", "fit", "export", "audit", "refit", "confirm", "all"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = prepare()
    if args.stage == "data":
        data(plan)
    if args.stage == "pilot":
        fit(plan, pilot=True)
    if args.stage in ("fit", "all"):
        fit(plan)
        release()
    if args.stage in ("export", "all"):
        export(plan)
        release()
    if args.stage in ("audit", "all"):
        if not evaluate(plan)["audit_gate_passed"]:
            print("LoRA candidate rejected by frozen audit gate; no final refit or validation300.", flush=True)
            return
        release()
    if args.stage in ("refit", "confirm", "all"):
        assert read_json(ROOT / "artifacts/improvement/lora_science_audit163_v1/metrics.json")["audit_gate_passed"]
    if args.stage in ("refit", "all"):
        fit(plan, final=True)
        release()
        export(plan, final=True)
        release()
    if args.stage in ("confirm", "all"):
        evaluate(plan, final=True)
        release()
        evaluate(plan, final=True, oracle=True)


if __name__ == "__main__":
    main()
