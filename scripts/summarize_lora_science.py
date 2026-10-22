"""Read-only paired quality, uncertainty, failures and selection-curve analysis."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from claim_checker.config import load_config
from claim_checker.data.grouped_training import claim_groups
from claim_checker.data.store import load_gold, read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import (
    LABELS,
    evidence_metrics,
    retrieval_metrics,
    verdict_metrics,
)
from scripts.summarize_polarity import matrix_scores

ROOT = Path(__file__).resolve().parents[1]


def plot_curve(points, epoch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in points]
    for ax, key, title, color in ((axes[0], "weighted_train_loss", "Gradient-fitting groups", "#2563eb"),
                                  (axes[1], "unweighted_stop_ce", "Separate stopping groups", "#b45309")):
        ax.plot(epochs, [r[key] for r in points], marker="o", color=color)
        ax.axvline(epoch, linestyle="--", color="#166534", alpha=.6)
        ax.set(title=title, xlabel="Epoch", ylabel="Classification objective (lower is better)", xticks=epochs)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"All-layer LoRA: selected epoch {epoch}")
    fig.text(.5, .01, "Training and stopping use different weights; compare trends within each panel.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for extension in ("png", "svg"):
        fig.savefig(ROOT / f"Docs/figures/lora_science_learning_curve.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def error_category(row):
    if row["status"] != "ok":
        return "operational error"
    if row["gold_label"] == "NEI":
        return "decisive verdict on gold NEI"
    docs = set(row["evidence_doc_ids"])
    for key, name in (("retrieved_doc_ids", "annotated paper absent from candidate pool"),
                      ("examined_doc_ids", "annotated paper not scored"),
                      ("reported_doc_ids", "annotated paper scored but not retained")):
        if not docs & set(row.get(key) or []):
            return name
    return "wrong verdict despite retained annotated paper"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit163", "validation300"), default="audit163")
    args = parser.parse_args()
    folder = ROOT / f"artifacts/improvement/lora_science_{args.stage}_v2"
    rows, report = read_jsonl(folder / "predictions.jsonl"), read_json(folder / "metrics.json")
    plan = read_json(ROOT / "artifacts/improvement/lora_science_protocol_v2.json")
    cfg = load_config()
    gold = load_gold(cfg.path("processed_dir"), "train" if args.stage == "audit163" else "validation")
    if args.stage == "audit163":
        baseline_paths = [ROOT / "artifacts/improvement/polarity_audit_v1/paired_predictions.jsonl"]
        baseline = {r["claim_id"]: r["control"] for r in read_jsonl(baseline_paths[0])}
        ids = set(plan["audit_ids"])
        group_map = read_json(ROOT / "artifacts/improvement/grouped_training_protocol_v1.json")["group_by_claim"]
    else:
        baseline_paths = [ROOT / f"artifacts/runs/{name}/predictions.jsonl"
                          for name in ("nli_titlefamily_recheck_v1", "nli_titlefamily_holdout_v1")]
        baseline = {r["claim_id"]: r for path in baseline_paths for r in read_jsonl(path)}
        ids = set(gold)
        group_map = claim_groups(list(gold.values()))
    assert len(rows) == len({r["claim_id"] for r in rows}) == len(ids)
    assert {r["claim_id"] for r in rows} == set(baseline) == ids
    for row in rows:
        assert row["gold_label"] == baseline[row["claim_id"]]["gold_label"] == gold[row["claim_id"]].label
        if row["status"] == "ok":
            initial = {h["doc_id"] for h in row["feedback_diagnostics"]["initial_retrieval_hits"]}
            assert set(row["retrieved_doc_ids"]) == initial
            assert set(row["examined_doc_ids"]) <= initial
            assert set(row["reported_doc_ids"]) <= set(row["examined_doc_ids"])
            assert len(row["reported_doc_ids"]) <= 3
    groups = sorted({group_map[r["claim_id"]] for r in rows})
    positions, codes = {g: i for i, g in enumerate(groups)}, {label: i for i, label in enumerate(LABELS)}
    matrices = {role: np.zeros((len(groups), 3, 4), dtype=np.int64) for role in ("baseline", "candidate")}
    metrics, correct = {}, {}
    for role in matrices:
        records = rows if role == "candidate" else [baseline[r["claim_id"]] for r in rows]
        correct[role] = []
        for row in records:
            predicted = row["verdict"] if row["status"] == "ok" else None
            matrices[role][positions[group_map[row["claim_id"]]], codes[row["gold_label"]], codes.get(predicted, 3)] += 1
            correct[role].append(predicted == row["gold_label"])
        verdict = verdict_metrics([r["gold_label"] for r in records], [r["verdict"] if r["status"] == "ok" else None for r in records])
        assert np.allclose(matrix_scores(matrices[role].sum(axis=0)), [verdict["accuracy"], verdict["macro_f1"]])
        metrics[role] = {"verdict": verdict, "retrieval": retrieval_metrics(records), "citations": evidence_metrics(records, "cited")}
    assert metrics["candidate"]["verdict"] == report["metrics"]["verdict"]
    draws = np.random.default_rng(42).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=10000)
    sampled = {role: matrix_scores(np.einsum("bg,gij->bij", draws, matrix)) for role, matrix in matrices.items()}
    intervals = {metric: np.percentile(sampled["candidate"][i] - sampled["baseline"][i], [2.5, 97.5]).tolist()
                 for i, metric in enumerate(("accuracy", "macro_f1"))}
    result = {"stage": args.stage, "claims": len(rows), "groups": len(groups), "metrics": metrics,
              "baseline_only_correct": sum(a and not b for a, b in zip(correct["baseline"], correct["candidate"], strict=True)),
              "candidate_only_correct": sum(b and not a for a, b in zip(correct["baseline"], correct["candidate"], strict=True)),
              "paired_group_bootstrap95_difference_candidate_minus_baseline": intervals,
              "bootstrap_draws": 10000, "seed": 42, "target": report["target"],
              "scored_source_budget": {"max": max(len(r["examined_doc_ids"]) for r in rows),
                                       "mean": float(np.mean([len(r["examined_doc_ids"]) for r in rows]))},
              "raw_predictions_sha256": sha256_file(folder / "predictions.jsonl"),
              "baseline_predictions_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in baseline_paths},
              "limitations": "Reused data, not blind. Group-bootstrap intervals do not correct selection. Several changes form a combined candidate. Hardware/runtime differs from saved baselines; timings are not a controlled latency comparison."}
    if "audit_gate_passed" in report:
        result["audit_gate_passed"] = report["audit_gate_passed"]
        result["gate_conditions"] = report["gate_conditions"]
    errors = [r for r in rows if not (r["status"] == "ok" and r["verdict"] == r["gold_label"])]
    result["error_categories"] = dict(Counter(error_category(r) for r in errors))
    with (folder / "failure_cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("claim_id", "claim", "gold", "predicted", "category", "rescue_sources", "quotes"))
        writer.writeheader()
        buckets = {label: iter(sorted((r for r in errors if r["gold_label"] == label),
                                     key=lambda r: r["claim_id"])) for label in ("CONTRADICT", "SUPPORT", "NEI")}
        cases = []
        while len(cases) < min(10, len(errors)):
            for bucket in buckets.values():
                row = next(bucket, None)
                if row is not None and len(cases) < 10:
                    cases.append(row)
        for row in cases:
            writer.writerow({"claim_id": row["claim_id"], "claim": row["claim"], "gold": row["gold_label"],
                             "predicted": row["verdict"], "category": error_category(row),
                             "rescue_sources": json.dumps(row["feedback_diagnostics"].get("rescued_doc_ids")),
                             "quotes": json.dumps(row["quotes"], ensure_ascii=False)})
    write_json(folder / "paired_quality.json", result)
    selected = read_json(ROOT / "artifacts/models/lora_science_select_v2/selection.json")
    curve = read_jsonl(ROOT / "artifacts/models/lora_science_select_v2/learning_curve.jsonl")
    plot_curve(curve, selected["epoch"])
    print({key: result[key] for key in ("claims", "baseline_only_correct", "candidate_only_correct", "paired_group_bootstrap95_difference_candidate_minus_baseline", "error_categories")})


if __name__ == "__main__":
    main()
