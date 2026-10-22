"""Development-only calibration; never accepts the final/holdout populations."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_jsonl, write_json
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.pipeline.aggregate import aggregate_verdicts


def decide(row, *, view, k, threshold):
    by_doc = {}
    for context in row["contexts"]:
        if context["rank"] > k or (view != "both" and context["view"] != view):
            continue
        scores = context["scores"]
        label = max(scores, key=scores.get)
        cutoff = threshold.get(label, 0.0) if isinstance(threshold, dict) else threshold
        if label != "NEI" and scores[label] < cutoff:
            label = "NEI"
        by_doc.setdefault(context["doc_id"], []).append(label)
    return aggregate_verdicts(label for labels in by_doc.values() for label in labels)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--asymmetric", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.folder / "predictions.jsonl")
    dev_ids = set(load_split(load_config(), "dev_tune")["claim_ids"])
    if not rows or any(r["claim_id"] not in dev_ids for r in rows):
        raise ValueError("Calibration is restricted to the development tuning split")
    results = []
    thresholds = [{"SUPPORT": s, "CONTRADICT": c} for s, c in itertools.product((0.5, 0.7, 0.9, 0.98), repeat=2)] if args.asymmetric else (0.5, 0.7, 0.8, 0.9)
    for view, k, threshold in itertools.product(("abstract", "selected", "both"), (1, 3), thresholds):
        settings = {"view": view, "k": k, "threshold": threshold}
        metrics = verdict_metrics([r["gold"] for r in rows], [decide(r, **settings) for r in rows])
        results.append({"settings": settings, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    results.sort(key=lambda r: (-r["macro_f1"], -r["accuracy"], json.dumps(r["settings"], sort_keys=True)))
    filename = "development_calibration_asymmetric.json" if args.asymmetric else "development_calibration.json"
    write_json(args.folder / filename, {"n": len(rows), "candidates": results})
    print(json.dumps(results[:8], indent=2))


if __name__ == "__main__":
    main()
