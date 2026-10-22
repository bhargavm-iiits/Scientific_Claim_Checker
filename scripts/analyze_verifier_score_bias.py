"""Additional development-only diagnostic; primary frozen confirmation is unchanged."""
import argparse
import itertools
import math
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import load_split
from claim_checker.data.store import read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.pipeline.aggregate import aggregate_verdicts

LABELS = ("SUPPORT", "CONTRADICT", "NEI")


def adjust_scores(scores, offsets):
    weighted = {label: scores[label] * math.exp(offsets.get(label, 0.0)) for label in LABELS}
    total = sum(weighted.values())
    return {label: value / total for label, value in weighted.items()}


def decide(row, offsets):
    labels = []
    for context in row["contexts"]:
        if context["view"] != "abstract" or context["rank"] > 3:
            continue
        scores = adjust_scores(context["scores"], offsets)
        label = max(scores, key=scores.get)
        labels.append(label if label == "NEI" or scores[label] >= 0.5 else "NEI")
    return aggregate_verdicts(labels)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    rows = read_jsonl(args.folder / "predictions.jsonl")
    expected = set(load_split(load_config(), "dev_tune")["claim_ids"])
    if {r["claim_id"] for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Exactly the complete internal development set is required")
    candidates = []
    for support, contradict in itertools.product((-1, 0, 1), (-4, -3, -2, -1, 0, 1)):
        offsets = {"SUPPORT": support, "CONTRADICT": contradict, "NEI": 0}
        metrics = verdict_metrics([r["gold"] for r in rows], [decide(r, offsets) for r in rows])
        candidates.append({"offsets": offsets, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    candidates.sort(key=lambda r: (-r["macro_f1"], -r["accuracy"], sum(abs(v) for v in r["offsets"].values())))
    report = {"purpose": "Secondary development-only class-score-adjustment diagnostic; does not replace the frozen primary confirmation",
              "population": "dev_tune120", "predictions_sha256": sha256_file(args.folder / "predictions.jsonl"),
              "candidates": candidates, "validation_used": False}
    write_json(args.folder / "secondary_score_bias_diagnostic.json", report)
    print(candidates[:5])


if __name__ == "__main__":
    main()
