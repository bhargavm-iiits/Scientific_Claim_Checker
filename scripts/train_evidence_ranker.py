"""Train evidence-aware ranking, audit it, and conditionally confirm on validation."""
import argparse
import gc
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.config import load_config, sha256_text
from claim_checker.data.evidence_reranking import labeled_windows
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
from claim_checker.evaluation.metrics import operational_metrics, quote_validity, verdict_metrics
from claim_checker.evaluation.runner import prediction_record, run_evaluation
from claim_checker.evaluation.training_provenance import assert_unfitted_population
from claim_checker.retrieval.index import Retriever, check_index
from claim_checker.retrieval.rerank import CrossEncoderScorer
from claim_checker.runtime import ClaimChecker
from scripts.refit_and_confirm_title_verifier import experiments_for

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/improvement/evidence_rank_protocol_v1.json"
SELECT = ROOT / "artifacts/models/evidence_rank_select_v1"
FINAL = ROOT / "artifacts/models/evidence_rank_final_v1"


def fingerprint(value):
    import json

    return sha256_text(json.dumps(value, sort_keys=True))


def candidates(cfg, corpus, gold, ids, plan):
    folder = ROOT / "artifacts/improvement/evidence_rank_candidates_v1"
    folder.mkdir(exist_ok=True)
    identity = {"index": check_index(cfg)["fingerprint"], "train_gold_sha256": plan["train_gold_sha256"],
                "retrieval_config": cfg.retrieval.model_dump(), "plan_sha256": sha256_file(PLAN),
                "code_sha256": sha256_file(Path(__file__))}
    manifest = folder / "identity.json"
    if manifest.exists():
        assert read_json(manifest) == identity, "Candidate identity changed"
    else:
        write_json(manifest, identity)
    path = folder / "candidates.jsonl"
    saved = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"]: r["doc_ids"] for r in saved}
    assert len(done) == len(saved) and set(done) <= set(plan["eligible_ids"])
    missing = [cid for cid in ids if cid not in done]
    if missing:
        retriever = Retriever.load(cfg, corpus)
        for index, cid in enumerate(missing, 1):
            output = retriever.candidates(gold[cid].claim, ("hybrid",))
            retrieved = [did for did, _ in output.candidates["hybrid"][:20]]
            append_jsonl(path, {"claim_id": cid, "doc_ids": retrieved})
            done[cid] = retrieved
            if index % 50 == 0:
                print(f"Generated annotation-free candidates {index}/{len(missing)}", flush=True)
        del retriever
        gc.collect()
    return {cid: done[cid] for cid in ids}


def build_rows(cfg, tokenizer, corpus, gold, ids, retrieved, forbidden_docs, plan):
    raw = {str(r["id"]): r for r in read_jsonl(cfg.path("raw_dir") / "scifact/claims_train.jsonl")}
    rows, omissions = [], []
    for cid in ids:
        claim = gold[cid]
        assert claim.split == "train" and not claim.conflict
        assert not set(claim.reference_doc_ids) & forbidden_docs
        selected = list(claim.reference_doc_ids)
        negatives = [did for did in retrieved[cid] if did not in selected and did not in forbidden_docs]
        selected += negatives[:plan["recipe"]["hard_negative_documents"]]
        for did in selected:
            label = claim.label_by_doc.get(did, "NEI") if did in claim.reference_doc_ids else "NEI"
            alternatives = [a["sentences"] for a in raw[cid]["evidence"].get(did, [])
                            if a["label"] == label and label != "NEI"]
            if label != "NEI":
                assert alternatives, "Decisive reference is missing original rationale annotations"
            windows, skipped = labeled_windows(tokenizer, claim.claim, corpus[did],
                                               reference=did in claim.reference_doc_ids, alternatives=alternatives)
            rows.extend({"claim_id": cid, **row} for row in windows)
            omissions.extend({"claim_id": cid, "doc_id": did, **item} for item in skipped)
            if alternatives and not any(row["target"] for row in windows):
                omissions.append({"claim_id": cid, "doc_id": did, "reason": "no_complete_rationale_in_runtime_windows"})
    return rows, omissions


def train(plan, final=False):
    cfg, recipe = load_config(), plan["recipe"]
    assert sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl") == plan["train_gold_sha256"]
    folder = FINAL if final else SELECT
    base = Path(snapshot_download(plan["base"], revision=plan["revision"], local_files_only=True))
    selected = read_json(SELECT / "selection.json") if final else None
    if final:
        assert read_json(ROOT / "artifacts/improvement/evidence_rank_audit_v1/metrics.json")["audit_gate_passed"]
        assert selected["weights_sha256"] == sha256_file(SELECT / f"epoch{selected['epoch']}/model.safetensors")
    identity = {"plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
                "window_builder_sha256": sha256_file(ROOT / "src/claim_checker/data/evidence_reranking.py"),
                "runtime_windows_sha256": sha256_file(ROOT / "src/claim_checker/retrieval/dense.py"),
                "base_weights_sha256": sha256_file(base / "model.safetensors"), "base_revision": plan["revision"],
                "final_fit": final, "selection_sha256": sha256_file(SELECT / "selection.json") if final else None}
    if folder.exists():
        assert read_json(folder / "identity.json") == identity, "Preserve the frozen training identity"
        assert (folder / "status.json").exists(), "Interrupted training; preserve it and use a new experiment version"
        return folder
    corpus = load_corpus(cfg.path("processed_dir"))
    gold, validation = load_gold(cfg.path("processed_dir"), "train"), load_gold(cfg.path("processed_dir"), "validation")
    forbidden_texts = {normalize_claim_text(g.claim) for g in validation.values()}
    fitting = plan["eligible_ids"] if final else plan["fit_ids"]
    stopping = [] if final else plan["stop_ids"]
    assert all(normalize_claim_text(gold[c].claim) not in forbidden_texts for c in fitting + stopping)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    retrieved = candidates(cfg, corpus, gold, fitting + stopping, plan)
    protected = {did for cid in plan["stop_ids"] + plan["audit_ids"] for did in gold[cid].reference_doc_ids} if not final else set()
    rows, omissions = build_rows(cfg, tokenizer, corpus, gold, fitting, retrieved, protected, plan)
    fit_docs = {row["doc_id"] for row in rows}
    stop_protected = fit_docs | {did for cid in plan["audit_ids"] for did in gold[cid].reference_doc_ids}
    stop_rows, stop_omissions = build_rows(cfg, tokenizer, corpus, gold, stopping, retrieved, stop_protected, plan)
    assert not fit_docs & {row["doc_id"] for row in stop_rows}
    assert {r["claim_id"] for r in rows} == set(fitting)
    assert {r["claim_id"] for r in stop_rows} == set(stopping)
    original = {str(r["id"]): r for r in read_jsonl(cfg.path("raw_dir") / "scifact/claims_train.jsonl")}
    for row in rows + stop_rows:
        doc = corpus[row["doc_id"]]
        item = gold[row["claim_id"]]
        assert row["claim"] == item.claim
        assert row["premise"] == f"{doc.title}\n{' '.join(doc.sentences[i] for i in row['sentence_ids'])}".strip()
        label = item.label_by_doc.get(doc.doc_id, "NEI") if doc.doc_id in item.reference_doc_ids else "NEI"
        alternatives = [a["sentences"] for a in original[item.claim_id]["evidence"].get(doc.doc_id, [])
                        if label != "NEI" and a["label"] == label]
        expected = int(any(set(rationale) <= set(row["sentence_ids"]) for rationale in alternatives if rationale))
        assert row["target"] == expected, "Ranking target differs from the original source annotation"
    folder.mkdir(parents=True)
    write_json(folder / "identity.json", identity)
    write_jsonl(folder / "training_windows.jsonl", rows)
    write_jsonl(folder / "stop_windows.jsonl", stop_rows)
    audit = {"fit_claims": len(fitting), "stop_claims": len(stopping), "fit_windows": len(rows), "stop_windows": len(stop_rows),
             "fit_targets": dict(Counter(r["target"] for r in rows)), "fit_stop_source_paper_overlap": 0,
             "validation_text_overlap": 0, "source_mismatches": 0, "training_omissions": omissions, "stop_omissions": stop_omissions,
             "supervision": dict(Counter(r["supervision"] for r in rows)), "all_evaluation_claims_retained": True}
    write_json(folder / "data_audit.json", audit)
    write_json(folder / "training_manifest.json", {"settings": {"final_fit": final}, "fitted_claim_ids": sorted(fitting),
               "fitted_claim_text_sha256": sorted({sha256_text(normalize_claim_text(gold[c].claim)) for c in fitting}),
               "training_data_sha256": sha256_file(folder / "training_windows.jsonl"), "protocol_sha256": sha256_file(PLAN),
               "note": "Evidence-window targets include explicitly weak negatives; no validation labels used"})
    print(f"Ranking data: {len(fitting)}fit/{len(stopping)}stop claims; {len(rows)}fit/{len(stop_rows)}stop windows; targets={audit['fit_targets']}", flush=True)
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    config.hidden_dropout_prob = recipe["dropout"]
    config.classifier_dropout = recipe["dropout"]
    config.sbert_ce_default_activation_function = "torch.nn.modules.linear.Identity"
    model = AutoModelForSequenceClassification.from_pretrained(base, config=config, local_files_only=True, dtype=torch.float32).cuda()
    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad = False

    def encode(values):
        return [(tokenizer(r["claim"], r["premise"]), r["target"]) for r in values]

    def collate(batch):
        tokens = tokenizer.pad([r[0] for r in batch], return_tensors="pt")
        return {k: v.cuda() for k, v in tokens.items()}, torch.tensor([r[1] for r in batch], dtype=torch.float32, device="cuda")

    loader = DataLoader(encode(rows), batch_size=recipe["physical_batch"], shuffle=True, collate_fn=collate)
    stop_loader = DataLoader(encode(stop_rows), batch_size=recipe["physical_batch"], collate_fn=collate)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    accumulation = recipe["effective_batch"] // recipe["physical_batch"]
    horizon = math.ceil(len(loader) / accumulation) * recipe["maximum_epochs"]
    warmup = max(1, horizon // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0.0, (horizon - step) / (horizon - warmup))))
    epochs = selected["epoch"] if final else recipe["maximum_epochs"]
    best_loss, best_epoch, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, (inputs, targets) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.flatten().float()
                per_example = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                start = index // accumulation * recipe["effective_batch"]
                loss = per_example.sum() / min(recipe["effective_batch"], len(rows) - start)
            assert torch.isfinite(loss), "Nonfinite ranking loss"
            loss.backward()
            losses.extend(per_example.detach().tolist())
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        if final:
            append_jsonl(folder / "learning_curve.jsonl", {"epoch": epoch, "train_bce": float(np.mean(losses))})
            print(f"Final ranker epoch{epoch}: trainBCE={np.mean(losses):.6f}", flush=True)
            continue
        model.eval()
        stop_losses = []
        for inputs, targets in stop_loader:
            with torch.inference_mode():
                logits = model(**inputs).logits.flatten().float()
                stop_losses.extend(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").tolist())
        stop_loss = float(np.mean(stop_losses))
        improved = stop_loss < best_loss - recipe["min_delta"]
        if improved:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            model.save_pretrained(folder / f"epoch{epoch}")
            tokenizer.save_pretrained(folder / f"epoch{epoch}")
        else:
            stale += 1
        row = {"epoch": epoch, "train_bce": float(np.mean(losses)), "stop_bce": stop_loss,
               "checkpoint_improved": improved, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
               "note": "Window-level component BCE with weak negatives; not claim accuracy"}
        append_jsonl(folder / "learning_curve.jsonl", row)
        print(row, flush=True)
        if stale >= recipe["patience"]:
            break
    if final:
        best_epoch = epochs
        model.save_pretrained(folder / f"epoch{epochs}")
        tokenizer.save_pretrained(folder / f"epoch{epochs}")
    write_json(folder / "selection.json", {"epoch": best_epoch, "stop_bce": None if final else best_loss,
               "weights_sha256": sha256_file(folder / f"epoch{best_epoch}/model.safetensors"), "validation_used": False})
    write_json(folder / "status.json", {"complete": True, "final_fit": final})
    del model, optimizer, parameters, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return folder


def profile(plan, ranker, reader):
    settings = plan["inference"]
    return {"product": {"judge_backend": "nli", "evidence_mode": "full_abstract", "prompt_set": "v1", "judge_think": False},
            "retrieval": {"reranker_model": str(ranker), "pipeline_k": settings["k"]},
            "verifier": {"model_dir": reader, "device": "cuda", "include_title": True, "window_overlap": 2,
                         "min_retrieval_score": settings["gate"], "support_threshold": 0.5, "contradict_threshold": 0.9}}


def audit_application(plan):
    selection = read_json(SELECT / "selection.json")
    ranker = SELECT / f"epoch{selection['epoch']}"
    assert selection["weights_sha256"] == sha256_file(ranker / "model.safetensors")
    assert plan["audit_reader_sha256"] == sha256_file(ROOT / plan["audit_reader"] / "model.safetensors")
    cfg = load_config(overrides=profile(plan, ranker, plan["audit_reader"]))
    gold = load_gold(cfg.path("processed_dir"), "train")
    claims = [gold[c] for c in plan["audit_ids"]]
    assert_unfitted_population(read_json(SELECT / "training_manifest.json"), claims)
    nli_pairs = read_jsonl((ROOT / plan["audit_reader"]).parent / "training_pairs.jsonl")
    assert_unfitted_population({"fitted_claim_ids": sorted({r["claim_id"] for r in nli_pairs})}, claims)
    folder = ROOT / "artifacts/improvement/evidence_rank_audit_v1"
    folder.mkdir(exist_ok=True)
    identity = {"plan_sha256": sha256_file(PLAN), "ranker_weights_sha256": selection["weights_sha256"],
                "reader_sha256": plan["audit_reader_sha256"], "configuration": cfg.model_dump(mode="json"),
                "index_fingerprint": check_index(cfg)["fingerprint"], "population": plan["audit_ids"],
                "script_sha256": sha256_file(Path(__file__)),
                "inference_code_sha256": {str(p.relative_to(ROOT / "src/claim_checker")): sha256_file(p)
                                          for p in sorted((ROOT / "src/claim_checker").rglob("*.py"))}}
    manifest = folder / "identity.json"
    if manifest.exists():
        assert read_json(manifest) == identity, "Changed application audit identity"
    else:
        write_json(manifest, identity)
    path = folder / "predictions.jsonl"
    saved = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in saved}
    assert len(done) == len(saved) and done <= set(plan["audit_ids"])
    # Explicitly verify the saved CrossEncoder keeps raw logits rather than sigmoid scores.
    scorer = CrossEncoderScorer(str(ranker), batch_size=8)
    assert isinstance(scorer.model.activation_fn, torch.nn.Identity), "Ranking gate requires raw logits"
    del scorer
    checker = ClaimChecker(cfg)
    for index, claim in enumerate(claims, 1):
        if claim.claim_id in done:
            continue
        result = checker.check(claim.claim, claim_id=claim.claim_id)
        append_jsonl(path, prediction_record("evidence_rank_audit", claim, result))
        if index % 20 == 0:
            print(f"Actual TRAIN application audit {index}/{len(claims)}", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == len(claims)
    metrics = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
    baseline_path = ROOT / plan["audit_baseline_path"]
    assert sha256_file(baseline_path) == plan["audit_baseline_sha256"]
    baseline = read_json(baseline_path)["metrics"]["verdict"]
    operations = operational_metrics(rows)
    passed = metrics["accuracy"] >= baseline["accuracy"] + 0.02 and metrics["macro_f1"] >= baseline["macro_f1"] + 0.02 and operations["errors"] == 0
    report = {"population": "163held-out originalTRAINclaims; notvalidation300", "verdict": metrics, "baseline": baseline,
              "operational": operations, "quotes": quote_validity(rows, checker.corpus), "audit_gate_passed": passed,
              "validation300_used": False, "recipe_changes_after_audit": False}
    write_json(folder / "metrics.json", report)
    print(f"Audit: {metrics['accuracy']:.4%}/{metrics['macro_f1']:.6f}; gatepassed={passed}", flush=True)
    del checker
    gc.collect()
    torch.cuda.empty_cache()
    return passed


def confirm(plan):
    selected = read_json(FINAL / "selection.json")
    ranker = FINAL / f"epoch{selected['epoch']}"
    assert selected["weights_sha256"] == sha256_file(ranker / "model.safetensors")
    assert plan["final_reader_sha256"] == sha256_file(ROOT / plan["final_reader"] / "model.safetensors")
    values = profile(plan, ranker, plan["final_reader"])
    values["verifier"]["device"] = "cpu"
    path = ROOT / "configs/evidence_rank_candidate.yaml"
    if path.exists():
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == values
    else:
        path.write_text("# Experimental learned evidence ranking; use measured validation results.\n" + yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    cfg = load_config(path, overrides={"verifier": {"device": "cuda"}})
    experiments = experiments_for(3)
    checker = ClaimChecker(cfg, experiments=experiments)
    combined, subsets = [], []
    for cohort in ("holdout", "recheck"):
        folder = run_evaluation(cfg, experiments, checker, suite=f"title_{cohort}", run_id=f"evidence_rank_{cohort}_v1")
        rows = read_jsonl(folder / "predictions.jsonl")
        combined.extend(rows)
        metrics = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
        subsets.append({"cohort": cohort, "metrics": metrics})
        print(f"Validation {cohort}: {metrics['accuracy']:.4%}/{metrics['macro_f1']:.6f}", flush=True)
    assert len(combined) == len({r["claim_id"] for r in combined}) == 300
    metrics = verdict_metrics([r["gold_label"] for r in combined], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in combined])
    report = {"pooled_validation_300": metrics, "subsets": subsets, "operational": operational_metrics(combined),
              "quotes": quote_validity(combined, checker.corpus), "population_note": "Unchanged reused validation300; notblind",
              "target_met": metrics["accuracy"] > 0.95 and metrics["macro_f1"] > 0.95, "protocol_sha256": sha256_file(PLAN)}
    write_json(ROOT / "artifacts/improvement/final_evidence_rank_results.json", report)
    print(f"Pooled300: {metrics['accuracy']:.4%}/{metrics['macro_f1']:.6f}; target_met={report['target_met']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("select", "audit", "refit", "confirm", "all"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = read_json(PLAN)
    if args.stage in ("select", "all"):
        train(plan)
    if args.stage in ("audit", "all"):
        if not audit_application(plan):
            print("Ranking candidate rejected before validation300; no final refit or validation run.", flush=True)
            return
    if args.stage in ("refit", "all"):
        train(plan, final=True)
    if args.stage in ("confirm", "all"):
        confirm(plan)


if __name__ == "__main__":
    main()
