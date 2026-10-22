"""Compare all three selection checkpoints on the complete frozen development set."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from claim_checker.data.store import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-name", default="qwen25_scifact_lora_train_v1")
    args = p.parse_args()
    training = ROOT / "artifacts/models" / args.training_name
    manifest = read_json(training / "training_manifest.json")
    if manifest["settings"]["final_fit"] or manifest["settings"]["max_steps"]:
        raise ValueError("Checkpoint selection requires a complete development-excluding training run")
    if not read_json(training / "status.json")["complete"]:
        raise ValueError("Training is incomplete")
    subprocess.run([sys.executable, "-X", "utf8", "scripts/audit_lora_training.py", str(training)], cwd=ROOT, check=True)
    results = []
    for epoch in range(1, manifest["settings"]["epochs"] + 1):
        name = f"{args.training_name}_epoch{epoch}_dev120"
        subprocess.run([sys.executable, "-X", "utf8", "scripts/benchmark_lora_judge.py",
                        "--adapter", f"artifacts/models/{args.training_name}/epoch{epoch}",
                        "--name", name, "--size", "120", "--max-length", str(manifest["settings"]["max_length"])],
                       cwd=ROOT, check=True)
        folder = ROOT / "artifacts/improvement" / name
        metrics = read_json(folder / "metrics.json")
        results.append({"epoch": epoch, "run": name, "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"], "metrics_sha256": sha256_file(folder / "metrics.json")})
    chosen = max(results, key=lambda r: (r["macro_f1"], r["accuracy"], -r["epoch"]))
    write_json(ROOT / "artifacts/improvement/frozen_selection_lora_v1.json", {
        "training_name": args.training_name, "selection_population": "dev_tune (120 claims)",
        "rule": "Macro-F1, then accuracy, then earlier epoch", "results": results, "chosen": chosen,
        "training_manifest_sha256": sha256_file(training / "training_manifest.json"),
        "plan_sha256": sha256_file(ROOT / "artifacts/improvement/lora_selection_plan.json"),
        "validation_used_for_selection": False,
    })
    print(f"Frozen development selection: {chosen}", flush=True)


if __name__ == "__main__":
    main()
