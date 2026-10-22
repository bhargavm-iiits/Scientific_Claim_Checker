"""Refine cached evidence-reader cutoffs on development only, with no model calls."""
import itertools
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import verdict_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "artifacts/improvement/trained_evidence_deberta_dev120_v2"
    coarse = read_json(folder / "selection.json")
    rows = {r["claim_id"]: r for r in read_jsonl(folder / "predictions.jsonl")}
    ids = load_split(load_config(), "dev_tune")["claim_ids"]
    assert set(rows) == set(ids) and read_json(folder / "status.json")["complete"]
    keys = ("alpha", "selector_threshold", "max_sentences", "k", "retrieval_gate")
    configurations, seen = [], set()
    for candidate in coarse["candidates"]:
        key = tuple(candidate[k] for k in keys)
        if key not in seen:
            configurations.append(dict(zip(keys, key, strict=True)))
            seen.add(key)
        if len(configurations) == 10:
            break
    plan_path = ROOT / "artifacts/improvement/evidence_reader_refinement_plan.json"
    plan = read_json(plan_path)
    candidates = []
    for cfg in configurations:
        maxima = []
        for cid in ids:
            support, contradict = 0.0, 0.0
            for doc in rows[cid]["documents"]:
                if doc["rank"] > cfg["k"] or (cfg["retrieval_gate"] is not None and doc["retrieval_score"] < cfg["retrieval_gate"]):
                    continue
                context = next(c for c in doc["contexts"] if (c["alpha"], c["threshold"], c["maximum"]) ==
                               (cfg["alpha"], cfg["selector_threshold"], cfg["max_sentences"]))
                for scores in context["window_scores"]:
                    label = max(scores, key=scores.get)
                    if label == "SUPPORT":
                        support = max(support, scores[label])
                    elif label == "CONTRADICT":
                        contradict = max(contradict, scores[label])
            maxima.append((support, contradict, rows[cid]["error"]))
        for support, contradict in itertools.product(plan["reader_cutoffs"], repeat=2):
            predictions = []
            for s, c, error in maxima:
                has_s, has_c = s >= support, c >= contradict
                predictions.append("ERROR" if error else "NEI" if has_s == has_c else "SUPPORT" if has_s else "CONTRADICT")
            metrics = verdict_metrics([rows[cid]["gold"] for cid in ids], predictions)
            candidates.append({**cfg, "support_threshold": support, "contradict_threshold": contradict,
                               "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    candidates.sort(key=lambda c: (-c["macro_f1"], -c["accuracy"]))
    chosen = candidates[0]
    report = {"chosen": chosen, "candidates": candidates, "population": "dev_tune120", "validation_used": False,
              "plan_sha256": sha256_file(plan_path), "source_predictions_sha256": sha256_file(folder / "predictions.jsonl"),
              "coarse_result": coarse["chosen"],
              "integration_criterion_passed": chosen["accuracy"] > 0.85 and chosen["macro_f1"] > 0.85}
    write_json(folder / "reader_calibration_refinement.json", report)
    print(chosen, flush=True)


if __name__ == "__main__":
    main()
