"""Read-only paired quality/uncertainty analysis for completed polarity runs."""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]


def matrix_scores(matrix):
    """Three fixed labels, with operational errors as a fourth prediction column."""
    true_positive = np.diagonal(matrix[..., :3], axis1=-2, axis2=-1)
    gold_count = matrix.sum(axis=-1)
    predicted_count = matrix[..., :3].sum(axis=-2)
    denominator = gold_count + predicted_count
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive, dtype=float), where=denominator != 0)
    return true_positive.sum(axis=-1) / matrix.sum(axis=(-2, -1)), f1.mean(axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen_isolated_gpu", "audit", "validation300"), required=True)
    args = parser.parse_args()
    folder = ROOT / f"artifacts/improvement/polarity_{args.stage}_v1"
    records = read_jsonl(folder / "paired_predictions.jsonl")
    metrics = read_json(folder / "metrics.json")
    assert len(records) == len({r["claim_id"] for r in records}) == metrics["claims"]
    cfg = load_config()
    gold = load_gold(cfg.path("processed_dir"), "validation" if args.stage == "validation300" else "train")
    groups = (claim_groups(list(gold.values())) if args.stage == "validation300" else
              read_json(ROOT / "artifacts/improvement/grouped_training_protocol_v1.json")["group_by_claim"])
    group_ids = sorted({groups[r["claim_id"]] for r in records})
    positions = {gid: i for i, gid in enumerate(group_ids)}
    codes = {label: i for i, label in enumerate(LABELS)}
    matrices = {role: np.zeros((len(group_ids), 3, 4), dtype=np.int64) for role in ("control", "candidate")}
    correct = {role: [] for role in matrices}
    quality = {}
    for row in records:
        assert row["control"]["gold_label"] == row["candidate"]["gold_label"] == gold[row["claim_id"]].label
        for role in matrices:
            r = row[role]
            predicted = r["verdict"] if r["status"] == "ok" else None
            matrices[role][positions[groups[row["claim_id"]]], codes[r["gold_label"]], codes.get(predicted, 3)] += 1
            correct[role].append(predicted == r["gold_label"])
    for role, matrix in matrices.items():
        rows = [r[role] for r in records]
        actual = verdict_metrics([r["gold_label"] for r in rows], [r["verdict"] if r["status"] == "ok" else None for r in rows])
        assert actual == metrics[role]["verdict"]
        assert np.allclose(matrix_scores(matrix.sum(axis=0)), [actual["accuracy"], actual["macro_f1"]])
        quality[role] = {"retrieval": retrieval_metrics(rows), "citations": evidence_metrics(rows, "cited"),
                         "quotes": metrics[role]["quotes"], "operations": metrics[role]["operations"]}
    rng = np.random.default_rng(42)
    draws = rng.multinomial(len(group_ids), np.full(len(group_ids), 1 / len(group_ids)), size=10000)
    sampled = {role: matrix_scores(np.einsum("bg,gij->bij", draws, matrix)) for role, matrix in matrices.items()}
    intervals = {}
    for index, metric in enumerate(("accuracy", "macro_f1")):
        delta = sampled["candidate"][index] - sampled["control"][index]
        intervals[f"{metric}_difference_candidate_minus_control"] = np.percentile(delta, [2.5, 97.5]).tolist()
    only_control = sum(a and not b for a, b in zip(correct["control"], correct["candidate"], strict=True))
    only_candidate = sum(b and not a for a, b in zip(correct["control"], correct["candidate"], strict=True))
    report = {"stage": args.stage, "claims": len(records), "paper_text_groups": len(group_ids),
              "bootstrap": {"unit": "Connected reference-paper / duplicate-text groups; paired group resampling",
                            "samples": 10000, "seed": 42, "fixed_macro_labels": LABELS, "percentile95": intervals},
              "only_control_correct": only_control, "only_candidate_correct": only_candidate,
              "quality": quality, "metrics": metrics,
              "limitations": "Reused internal development/validation data, not blind. Intervals do not undo selection or establish absence of overfitting.",
              "source_sha256": sha256_file(folder / "paired_predictions.jsonl")}
    cpu_path = ROOT / "artifacts/improvement/polarity_screen_v1/paired_predictions.jsonl"
    if args.stage == "screen_isolated_gpu" and cpu_path.exists():
        cpu = {r["claim_id"]: r for r in read_jsonl(cpu_path)}
        shared = [r for r in records if r["claim_id"] in cpu]
        report["partial_cpu_gpu_parity"] = {
            "claims": len(shared), "selection": "Available prefix from interrupted CPU run, not a random parity sample",
            "verdict_matches": sum(r[role]["verdict"] == cpu[r["claim_id"]][role]["verdict"]
                                   for r in shared for role in ("control", "candidate")),
            "comparisons": 2 * len(shared)}
    write_json(folder / "paired_quality.json", report)
    print({"claims": len(records), "groups": len(group_ids), "control_only_correct": only_control,
           "candidate_only_correct": only_candidate, "bootstrap95": intervals}, flush=True)


if __name__ == "__main__":
    main()
