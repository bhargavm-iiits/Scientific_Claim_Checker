"""Hardware pilots, training and development-only selection of the larger verifier."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil

from claim_checker.data.store import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/improvement/nli_large_selection_plan.json"


def run(arguments, log):
    command = [sys.executable, "-X", "utf8", *arguments]
    print("Starting:", " ".join(arguments), flush=True)
    with log.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="nli_large_scifact_train_v1")
    args = p.parse_args()
    folder = ROOT / "artifacts/models" / args.name
    state_path = ROOT / f"artifacts/improvement/{args.name}_workflow.json"
    log = ROOT / f"artifacts/improvement/{args.name}_workflow.log"
    # Do not compete with the separately running generative checkpoint evaluation.
    while True:
        active = [process.pid for process in psutil.process_iter(["cmdline"])
                  if any(s.replace("\\", "/").rsplit("/", 1)[-1] in
                         {"benchmark_lora_judge.py", "score_lora_checkpoints.py"}
                         for s in (process.info["cmdline"] or []))]
        if not active:
            break
        print(f"Waiting for LoRA evaluation processes: {active}", flush=True)
        time.sleep(5)
    base = ["--base-dir", "artifacts/models/nli_large_base", "--epochs", "4", "--layers", "12",
            "--max-length", "512", "--learning-rate", "1e-5", "--hard-negatives", "2",
            "--device", "cuda", "--amp", "--gradient-checkpointing", "--global-class-balance"]
    write_json(state_path, {"phase": "hardware_pilots", "plan_sha256": sha256_file(PLAN)})
    pilots = []
    for batch in (1, 2):
        name = f"{args.name}_pilot_b{batch}"
        pilot_path = ROOT / "artifacts/models" / name / "pilot.json"
        if not pilot_path.exists():
            run(["scripts/train_verifier.py", "--name", name, *base, "--batch-size", str(batch),
                 "--accumulate", str(8 // batch), "--pilot-batches", str(32 // batch)], log)
        stats = read_json(pilot_path)
        pilots.append({"batch_size": batch, "accumulate": 8 // batch,
                       "seconds_per_example": stats["seconds"] / 32, **stats})
    feasible = [r for r in pilots if r["gpu_peak_reserved_gib"] <= 5.2]
    if not feasible:
        raise MemoryError(f"No pilot fits the declared memory allowance: {pilots}")
    hardware = min(feasible, key=lambda r: r["seconds_per_example"])
    write_json(ROOT / f"artifacts/improvement/{args.name}_hardware.json", {"pilots": pilots, "chosen": hardware})
    print(f"Hardware selected without evaluation labels: {hardware}", flush=True)
    status = folder / "status.json"
    if not status.exists() or not read_json(status)["complete"]:
        write_json(state_path, {"phase": "training", "hardware": hardware})
        run(["scripts/train_verifier.py", "--name", args.name, *base, "--batch-size", str(hardware["batch_size"]),
             "--accumulate", str(hardware["accumulate"])], log)
    run(["scripts/verify_training_artifact.py", str(folder)], log)
    write_json(state_path, {"phase": "development_scoring", "hardware": hardware})
    run(["scripts/score_verifier_checkpoints.py", "--training-run", args.name, "--epochs", "1", "2", "3", "4"], log)
    candidates = []
    for epoch in range(1, 5):
        dev = ROOT / f"artifacts/improvement/{args.name}_epoch{epoch}_dev120"
        run(["scripts/analyze_verifier.py", str(dev), "--asymmetric"], log)
        calibration = dev / "development_calibration_asymmetric.json"
        for candidate in read_json(calibration)["candidates"]:
            if candidate["settings"]["view"] == "abstract":
                candidates.append({"epoch": epoch, **candidate, "calibration_sha256": sha256_file(calibration)})
    candidates.sort(key=lambda c: (-c["macro_f1"], -c["accuracy"], c["epoch"], str(c["settings"])))
    frozen = {"training_name": args.name, "chosen": candidates[0], "candidates": candidates,
              "selection_population": "dev_tune120", "validation_used_for_selection": False,
              "plan_sha256": sha256_file(PLAN), "training_manifest_sha256": sha256_file(folder / "training_manifest.json"),
              "hardware": hardware}
    write_json(ROOT / "artifacts/improvement/frozen_selection_nli_large_v1.json", frozen)
    write_json(state_path, {"phase": "complete", "chosen": candidates[0]})
    print(f"Frozen development selection: {candidates[0]}", flush=True)


if __name__ == "__main__":
    main()
