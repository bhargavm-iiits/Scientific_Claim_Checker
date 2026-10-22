"""Separately stop and calibrate an evidence ranker before application auditing."""
import argparse
import gc
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from claim_checker.config import load_config, sha256_text
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
from claim_checker.retrieval.index import check_index
from claim_checker.retrieval.rerank import CrossEncoderScorer, rerank_documents
from claim_checker.runtime import ClaimChecker
from scripts.refit_and_confirm_title_verifier import experiments_for
from scripts.train_evidence_ranker import profile

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/improvement/evidence_rank_calibrated_protocol_v2.json"
MODEL = ROOT / "artifacts/models/evidence_rank_calibrated_v2"
AUDIT = ROOT / "artifacts/improvement/evidence_rank_calibrated_audit_v2"


def lower_tail_gate(scores, alpha=0.05):
    """A fixed finite-sample order statistic; not a claim-accuracy guarantee."""
    if not scores or not 0 < alpha < 1 or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Finite calibration scores and alpha in (0,1) are required")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return None if rank > len(scores) else float(sorted(scores)[len(scores) - rank])


def prepare():
    parent_path = ROOT / "artifacts/improvement/evidence_rank_protocol_v1.json"
    parent = read_json(parent_path)
    cfg = load_config()
    gold = load_gold(cfg.path("processed_dir"), "train")
    groups = read_json(ROOT / "artifacts/improvement/grouped_training_protocol_v1.json")["group_by_claim"]
    ids = parent["stop_ids"]
    split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=43)
    early_indices, calib_indices = next(split.split(ids, [gold[c].label for c in ids], [groups[c] for c in ids]))
    stopping, calibration = sorted(ids[i] for i in early_indices), sorted(ids[i] for i in calib_indices)
    assert not {groups[c] for c in stopping} & {groups[c] for c in calibration}
    plan = {**parent, "parent_protocol_sha256": sha256_file(parent_path),
            "stop_ids": stopping, "calibration_ids": calibration, "group_by_claim": groups,
            "alpha": 0.05, "calibration": "Lower-tail finite-sample order statistic on per-paper-group minimum annotated-positive document scores; no threshold grid",
            "source_data": "Reuse v1fittingwindows from484claims, initialize fresh genericweights; stoppingrows only stoppingIDs and exclude calibration reference papers",
            "deployment": "Keep selected ranker weights unchanged after calibration; NO807rankerrefit that would invalidate its learned score cutoff",
            "refit": "Disabled: keep the selected484TRAINranker weights unchanged after calibration",
            "development_note": "Chosen after aggregate v1TRAINaudit showed gated evidence loss. The same163TRAINaudit is now reused development evidence, not blind. No individual validation errors or labels choose this recipe.",
            "inference": {**parent["inference"], "gate": "computed from independent TRAIN calibration groups"}}
    if PLAN.exists():
        assert read_json(PLAN) == plan, "Preserve the frozen calibration recipe"
    else:
        write_json(PLAN, plan)
    print(f"Frozen{len(parent['fit_ids'])}fit/{len(stopping)}stop/{len(calibration)}calibration/{len(parent['audit_ids'])}audit; no807rankerrefit", flush=True)
    return plan


def register_used_claims(plan, calibration_path=None):
    cfg = load_config()
    gold = load_gold(cfg.path("processed_dir"), "train")
    used = sorted(set(plan["fit_ids"] + plan["stop_ids"] + (plan["calibration_ids"] if calibration_path else [])))
    write_json(MODEL / "training_manifest.json", {"settings": {"final_fit": False}, "gradient_fit_claim_ids": plan["fit_ids"],
               "stopping_claim_ids": plan["stop_ids"], "calibration_claim_ids": plan["calibration_ids"] if calibration_path else [],
               "fitted_claim_ids": used,
               "fitted_claim_text_sha256": sorted({sha256_text(normalize_claim_text(gold[c].claim)) for c in used}),
               "protocol_sha256": sha256_file(PLAN),
               "calibration_sha256": sha256_file(calibration_path) if calibration_path else None,
               "note": "Fitted IDs include every claim used for weights, checkpoint choice or score-cutoff calibration"})


def fit(plan):
    cfg, recipe = load_config(), plan["recipe"]
    source = ROOT / "artifacts/models/evidence_rank_select_v1"
    base = Path(snapshot_download(plan["base"], revision=plan["revision"], local_files_only=True))
    identity = {"plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
                "base_weights_sha256": sha256_file(base / "model.safetensors"),
                "source_fit_sha256": sha256_file(source / "training_windows.jsonl"),
                "source_stop_sha256": sha256_file(source / "stop_windows.jsonl"),
                "source_annotation_audit_sha256": sha256_file(source / "data_audit.json")}
    if MODEL.exists():
        assert read_json(MODEL / "identity.json") == identity and (MODEL / "status.json").exists(), "Preserve complete runs; do not mix changed/interrupted training"
        return
    original_manifest = read_json(source / "training_manifest.json")
    assert identity["source_fit_sha256"] == original_manifest["training_data_sha256"]
    gold = load_gold(cfg.path("processed_dir"), "train")
    calibration_papers = {d for c in plan["calibration_ids"] for d in gold[c].reference_doc_ids}
    rows = read_jsonl(source / "training_windows.jsonl")
    stop_rows = [r for r in read_jsonl(source / "stop_windows.jsonl") if r["claim_id"] in set(plan["stop_ids"]) and r["doc_id"] not in calibration_papers]
    assert {r["claim_id"] for r in rows} == set(plan["fit_ids"])
    assert {r["claim_id"] for r in stop_rows} == set(plan["stop_ids"])
    fit_docs, stop_docs = {r["doc_id"] for r in rows}, {r["doc_id"] for r in stop_rows}
    assert not fit_docs & (stop_docs | calibration_papers) and not stop_docs & calibration_papers
    assert not set(plan["fit_ids"]) & (set(plan["stop_ids"]) | set(plan["calibration_ids"]) | set(plan["audit_ids"]))
    assert not set(plan["stop_ids"]) & set(plan["calibration_ids"])
    forbidden = {normalize_claim_text(g.claim) for g in load_gold(cfg.path("processed_dir"), "validation").values()}
    assert all(normalize_claim_text(gold[c].claim) not in forbidden for c in plan["fit_ids"] + plan["stop_ids"] + plan["calibration_ids"])
    MODEL.mkdir(parents=True)
    write_json(MODEL / "identity.json", identity)
    write_jsonl(MODEL / "training_windows.jsonl", rows)
    write_jsonl(MODEL / "stop_windows.jsonl", stop_rows)
    write_json(MODEL / "data_audit.json", {"fit_claims": len(plan["fit_ids"]), "stop_claims": len(plan["stop_ids"]),
               "calibration_claims": len(plan["calibration_ids"]), "fit_windows": len(rows), "stop_windows": len(stop_rows),
               "fit_stop_calibration_source_paper_overlap": 0, "validation_text_overlap": 0,
               "source_fit_sha256": identity["source_fit_sha256"], "annotation_audit_inherited_from": str(source / "data_audit.json")})
    register_used_claims(plan)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
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
        return {k: v.cuda() for k, v in tokens.items()}, torch.tensor([r[1] for r in batch], device="cuda", dtype=torch.float32)

    loader = DataLoader(encode(rows), batch_size=8, shuffle=True, collate_fn=collate)
    stopping = DataLoader(encode(stop_rows), batch_size=8, collate_fn=collate)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    horizon = math.ceil(len(loader) / 4) * recipe["maximum_epochs"]
    warmup = max(1, horizon // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, max(0.0, (horizon - step) / (horizon - warmup))))
    best, best_epoch, stale = float("inf"), None, 0
    for epoch in range(1, 5):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (inputs, targets) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.flatten().float()
                values = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                loss = values.sum() / min(32, len(rows) - step // 4 * 32)
            assert torch.isfinite(loss)
            loss.backward()
            losses.extend(values.detach().tolist())
            if (step + 1) % 4 == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        stop_losses = []
        for inputs, targets in stopping:
            with torch.inference_mode():
                logits = model(**inputs).logits.flatten().float()
                stop_losses.extend(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").tolist())
        stop_loss = float(np.mean(stop_losses))
        improved = stop_loss < best - recipe["min_delta"]
        if improved:
            best, best_epoch, stale = stop_loss, epoch, 0
            model.save_pretrained(MODEL / f"epoch{epoch}")
            tokenizer.save_pretrained(MODEL / f"epoch{epoch}")
        else:
            stale += 1
        report = {"epoch": epoch, "train_bce": float(np.mean(losses)), "stop_bce": stop_loss, "checkpoint_improved": improved}
        append_jsonl(MODEL / "learning_curve.jsonl", report)
        print(report, flush=True)
        if stale >= recipe["patience"]:
            break
    write_json(MODEL / "selection.json", {"epoch": best_epoch, "stop_bce": best,
               "weights_sha256": sha256_file(MODEL / f"epoch{best_epoch}/model.safetensors"), "calibration_used": False, "validation_used": False})
    write_json(MODEL / "status.json", {"complete": True})
    del model, optimizer, parameters, scheduler
    gc.collect()
    torch.cuda.empty_cache()


def calibrate(plan):
    chosen = read_json(MODEL / "selection.json")
    ranker = MODEL / f"epoch{chosen['epoch']}"
    assert sha256_file(ranker / "model.safetensors") == chosen["weights_sha256"]
    path = MODEL / "calibration.json"
    if path.exists():
        result = read_json(path)
        assert result["weights_sha256"] == chosen["weights_sha256"] and result["plan_sha256"] == sha256_file(PLAN)
        register_used_claims(plan, path)
        return result
    cfg = load_config()
    gold, corpus = load_gold(cfg.path("processed_dir"), "train"), load_corpus(cfg.path("processed_dir"))
    scorer = CrossEncoderScorer(str(ranker), device="cuda", batch_size=16)
    assert isinstance(scorer.model.activation_fn, torch.nn.Identity)
    groups, pairs = {}, []
    for cid in plan["calibration_ids"]:
        claim = gold[cid]
        for did in claim.evidence_by_doc:
            if claim.label_by_doc.get(did, "NEI") == "NEI":
                continue
            ranked = rerank_documents(claim.claim, [corpus[did]], scorer.score, scorer.count_tokens)
            score = ranked[0][1]
            group = plan["group_by_claim"][cid]
            groups[group] = min(groups.get(group, score), score)
            pairs.append({"claim_id": cid, "doc_id": did, "group": group, "score": score})
    gate = lower_tail_gate(list(groups.values()), plan["alpha"])
    result = {"weights_sha256": chosen["weights_sha256"], "plan_sha256": sha256_file(PLAN), "gate": gate,
              "alpha": plan["alpha"], "positive_groups": len(groups), "positive_pairs": len(pairs),
              "group_minimum_scores": groups, "pairs": pairs,
              "meaning": "Targets retention of annotated evidence in TRAIN calibration, NOT95%claimaccuracy or a validated coverage guarantee"}
    write_json(path, result)
    register_used_claims(plan, path)
    print(f"Calibratedgate={gate}from{len(groups)}positiveTRAINgroups/{len(pairs)}pairs; this is NOTclaimaccuracy", flush=True)
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def configuration(plan, calibration, reader):
    chosen = read_json(MODEL / "selection.json")
    values = profile({**plan, "inference": {**plan["inference"], "gate": calibration["gate"]}},
                     MODEL / f"epoch{chosen['epoch']}", reader)
    return values


def audit(plan, calibration):
    cfg = load_config(overrides=configuration(plan, calibration, plan["audit_reader"]))
    assert sha256_file(ROOT / plan["audit_reader"] / "model.safetensors") == plan["audit_reader_sha256"]
    gold = load_gold(cfg.path("processed_dir"), "train")
    claims = [gold[c] for c in plan["audit_ids"]]
    assert_unfitted_population(read_json(MODEL / "training_manifest.json"), claims)
    reader_rows = read_jsonl((ROOT / plan["audit_reader"]).parent / "training_pairs.jsonl")
    assert_unfitted_population({"fitted_claim_ids": list({r["claim_id"] for r in reader_rows})}, claims)
    AUDIT.mkdir(exist_ok=True)
    identity = {"protocol_sha256": sha256_file(PLAN), "calibration_sha256": sha256_file(MODEL / "calibration.json"),
                "ranker_sha256": calibration["weights_sha256"], "configuration": cfg.model_dump(mode="json"),
                "index_fingerprint": check_index(cfg)["fingerprint"], "script_sha256": sha256_file(Path(__file__)),
                "core_code_sha256": {str(p.relative_to(ROOT / "src/claim_checker")): sha256_file(p)
                                     for p in sorted((ROOT / "src/claim_checker").rglob("*.py"))}}
    manifest = AUDIT / "identity.json"
    if manifest.exists():
        assert read_json(manifest) == identity
    else:
        write_json(manifest, identity)
    path = AUDIT / "predictions.jsonl"
    saved = read_jsonl(path) if path.exists() else []
    done = {r["claim_id"] for r in saved}
    assert len(done) == len(saved) and done <= set(plan["audit_ids"])
    checker = ClaimChecker(cfg)
    for index, claim in enumerate(claims, 1):
        if claim.claim_id not in done:
            result = checker.check(claim.claim, claim_id=claim.claim_id)
            append_jsonl(path, prediction_record("calibrated_rank_audit", claim, result))
        if index % 20 == 0:
            print(f"Calibrated realapplicationaudit {index}/{len(claims)}", flush=True)
    rows = read_jsonl(path)
    assert len(rows) == 163
    metrics = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
    baseline = read_json(ROOT / plan["audit_baseline_path"])["metrics"]["verdict"]
    operations = operational_metrics(rows)
    passed = metrics["accuracy"] >= baseline["accuracy"] + 0.02 and metrics["macro_f1"] >= baseline["macro_f1"] + 0.02 and operations["errors"] == 0
    write_json(AUDIT / "metrics.json", {"verdict": metrics, "baseline": baseline, "operational": operations,
               "quotes": quote_validity(rows, checker.corpus), "audit_gate_passed": passed, "validation300_used": False,
               "population": "Reused163TRAINaudit, notblind or validation300", "calibrated_gate": calibration["gate"]})
    print(f"Calibratedaudit: {metrics['accuracy']:.4%}/{metrics['macro_f1']:.6f}; gatepassed={passed}", flush=True)
    del checker
    gc.collect()
    torch.cuda.empty_cache()
    return passed


def confirm(plan, calibration):
    assert read_json(AUDIT / "metrics.json")["audit_gate_passed"]
    assert sha256_file(ROOT / plan["final_reader"] / "model.safetensors") == plan["final_reader_sha256"]
    values = configuration(plan, calibration, plan["final_reader"])
    values["verifier"]["device"] = "cpu"
    path = ROOT / "configs/evidence_rank_calibrated_candidate.yaml"
    if path.exists():
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == values
    else:
        path.write_text("# Experimental independently calibrated evidence ranker; use measured validation results.\n" + yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    cfg = load_config(path, overrides={"verifier": {"device": "cuda"}})
    experiments = experiments_for(3)
    checker = ClaimChecker(cfg, experiments=experiments)
    rows, subsets = [], []
    for cohort in ("holdout", "recheck"):
        folder = run_evaluation(cfg, experiments, checker, suite=f"title_{cohort}", run_id=f"calibrated_rank_{cohort}_v2")
        current = read_jsonl(folder / "predictions.jsonl")
        rows.extend(current)
        result = verdict_metrics([r["gold_label"] for r in current], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in current])
        subsets.append({"cohort": cohort, "metrics": result})
        print(f"Actual validation{cohort}: {result['accuracy']:.4%}/{result['macro_f1']:.6f}", flush=True)
    assert len(rows) == len({r["claim_id"] for r in rows}) == 300
    result = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
    write_json(ROOT / "artifacts/improvement/final_calibrated_rank_results.json", {"pooled_validation_300": result,
               "subsets": subsets, "operational": operational_metrics(rows), "quotes": quote_validity(rows, checker.corpus),
               "target_met": result["accuracy"] > 0.95 and result["macro_f1"] > 0.95,
               "population_note": "Unchanged reusedvalidation300, notblind; calibration evidence retention is not accuracy"})
    print(f"Actual validation300: {result['accuracy']:.4%}/{result['macro_f1']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "fit", "calibrate", "audit", "confirm", "all"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = prepare()
    if args.stage in ("fit", "all"):
        fit(plan)
    if args.stage in ("calibrate", "audit", "confirm", "all"):
        calibration = calibrate(plan)
    if args.stage in ("audit", "all"):
        if not audit(plan, calibration):
            print("Calibrated ranking candidate rejected; no validation300run.", flush=True)
            return
    if args.stage in ("confirm", "all"):
        confirm(plan, calibration)


if __name__ == "__main__":
    main()
