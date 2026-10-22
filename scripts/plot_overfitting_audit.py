"""Export saved training and held-out curves; do not run or tune models."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from claim_checker.data.store import read_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = read_json(ROOT / "artifacts/improvement/overfitting_audit.json")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.85, wspace=0.3, hspace=0.65)
    for row, (key, train_key, metric_key, title, metric_title) in enumerate((
        ("nli_learning_curve", "mean_training_loss", "fixed_setting_macro_f1", "NLI reader", "Claim Macro-F1 (fixed settings)"),
        ("selector_learning_curve", "mean_training_loss", "reference_dev_f1", "Rationale selector", "Sentence F1 (reference docs supplied)"),
    )):
        points = audit[key]
        epochs = [p["epoch"] for p in points]
        for col, (value, color, ylabel) in enumerate(((train_key, "#2563eb", "Training loss (lower is better)"),
                                                     (metric_key, "#b45309", "Development F1 (higher is better)"))):
            ax = axes[row, col]
            values = [p[value] for p in points]
            ax.plot(epochs, values, marker="o", color=color, linewidth=2)
            ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{title}\n{'Training' if col == 0 else metric_title}", xticks=epochs)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 1:
                best = max(points, key=lambda p: p[value])
                ax.axvline(best["epoch"], color="#166534", linestyle="--", alpha=0.7)
                ax.annotate(f"Selected epoch {best['epoch']}", (best["epoch"], best[value]), xytext=(8, -22),
                            textcoords="offset points", fontsize=9, color="#166534")
    fig.suptitle("Lower training loss did not produce better development scores", fontsize=14, y=0.97)
    folder = ROOT / "Docs/figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "overfitting_curves_2026-10-22.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(folder / "overfitting_curves_2026-10-22.svg", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("Saved PNG and SVG learning curves.")


if __name__ == "__main__":
    main()
