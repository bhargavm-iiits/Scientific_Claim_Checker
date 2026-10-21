"""Read-only paired analysis of the fixed joint-reader TRAIN audit."""
from pathlib import Path

import numpy as np

from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import (
    LABELS,
    evidence_metrics,
    retrieval_metrics,
    verdict_metrics,
)
from scripts.summarize_polarity import matrix_scores

ROOT = Path(__file__).resolve().parents[1]


def plot_learning_curve(points, selected_epoch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [p["epoch"] for p in points]
    for ax, key, title, color in (
        (axes[0], "weighted_train_loss", "Gradient-fitting groups", "#2563eb"),
        (axes[1], "unweighted_stop_joint_loss", "Separate stopping groups", "#b45309"),
    ):
        ax.plot(epochs, [p[key] for p in points], marker="o", color=color)
        ax.set(title=title, xlabel="Epoch", ylabel="Joint objective (lower is better)", xticks=epochs)
        ax.axvline(selected_epoch, color="#166534", linestyle="--", alpha=.6)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Joint reader: selected epoch {selected_epoch}")
    fig.text(.5, .01, "Training and stopping use different classification weights; compare trends within each panel.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    destination = ROOT / "Docs/figures"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(destination / f"joint_reader_learning_curve.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    folder = ROOT / "artifacts/improvement/joint_audit163_v1"
    rows = read_jsonl(folder / "predictions.jsonl")
    report = read_json(folder / "metrics.json")
    plan = read_json(ROOT / "artifacts/improvement/joint_nli_protocol_v1.json")
    assert len(rows) == len({r["claim_id"] for r in rows}) == 163
    assert {r["claim_id"] for r in rows} == set(plan["audit_ids"])
    baseline_file = ROOT / "artifacts/improvement/polarity_audit_v1/paired_predictions.jsonl"
    baseline = {r["claim_id"]: r["control"] for r in read_jsonl(baseline_file)}
    assert set(baseline) == set(plan["audit_ids"])
    group_by_claim = read_json(ROOT / "artifacts/improvement/grouped_training_protocol_v1.json")["group_by_claim"]
    groups = sorted({group_by_claim[r["claim_id"]] for r in rows})
    positions = {gid: i for i, gid in enumerate(groups)}
    codes = {label: i for i, label in enumerate(LABELS)}
    matrices = {role: np.zeros((len(groups), 3, 4), dtype=np.int64) for role in ("baseline", "joint")}
    correct = {role: [] for role in matrices}
    metrics = {}
    for role in matrices:
        records = rows if role == "joint" else [baseline[r["claim_id"]] for r in rows]
        for r in records:
            assert r["gold_label"] == baseline[r["claim_id"]]["gold_label"]
            predicted = r["verdict"] if r["status"] == "ok" else None
            matrices[role][positions[group_by_claim[r["claim_id"]]], codes[r["gold_label"]], codes.get(predicted, 3)] += 1
            correct[role].append(predicted == r["gold_label"])
        scores = verdict_metrics([r["gold_label"] for r in records],
                                 [r["verdict"] if r["status"] == "ok" else None for r in records])
        assert np.allclose(matrix_scores(matrices[role].sum(axis=0)), [scores["accuracy"], scores["macro_f1"]])
        metrics[role] = {"verdict": scores, "retrieval": retrieval_metrics(records),
                         "citations": evidence_metrics(records, "cited")}
    assert metrics["joint"]["verdict"] == report["verdict"]
    for key in ("accuracy", "macro_f1"):
        assert abs(metrics["baseline"]["verdict"][key] - plan["audit_baseline"][key]) < 1e-12
    draws = np.random.default_rng(42).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=10000)
    sampled = {role: matrix_scores(np.einsum("bg,gij->bij", draws, matrix)) for role, matrix in matrices.items()}
    intervals = {metric: np.percentile(sampled["joint"][i] - sampled["baseline"][i], [2.5, 97.5]).tolist()
                 for i, metric in enumerate(("accuracy", "macro_f1"))}
    learning = read_jsonl(ROOT / "artifacts/models/joint_nli_select_v1/learning_curve.jsonl")
    selected = read_json(ROOT / "artifacts/models/joint_nli_select_v1/selection.json")
    result = {"claims": len(rows), "paper_text_groups": len(groups), "metrics": metrics,
              "baseline_only_correct": sum(a and not b for a, b in zip(correct["baseline"], correct["joint"], strict=True)),
              "joint_only_correct": sum(b and not a for a, b in zip(correct["baseline"], correct["joint"], strict=True)),
              "paired_group_bootstrap95_difference_joint_minus_baseline": intervals,
              "bootstrap_samples": 10000, "seed": 42, "selection": selected, "learning_curve": learning,
              "gate_passed": report["audit_gate_passed"], "quotes": report["quotes"],
              "limitations": "Reused TRAIN audit, not blind. Intervals do not correct experiment selection. Baseline is the saved same-weight common-window control from polarity audit; model architecture, training objective and verdict cutoffs differ. Not a causal ablation or proof against overfitting.",
              "input_sha256": {"candidate": sha256_file(folder / "predictions.jsonl"), "baseline": sha256_file(baseline_file)}}
    write_json(folder / "paired_quality.json", result)
    plot_learning_curve(learning, selected["epoch"])
    print({"claims": len(rows), "gate_passed": result["gate_passed"], "joint": metrics["joint"]["verdict"],
           "paired_bootstrap95": intervals})


if __name__ == "__main__":
    main()
