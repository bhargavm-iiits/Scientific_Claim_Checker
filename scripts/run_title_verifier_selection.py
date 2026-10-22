"""Evidence-complete, all-encoder training and development-only model selection."""
import argparse
import itertools
import subprocess
from pathlib import Path

from analyze_verifier import decide
from run_large_verifier_selection import run

from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.metrics import verdict_metrics

ROOT = Path(__file__).resolve().parents[1]
NAME = "nli_large_title_train_v2"
PLAN = ROOT / "artifacts/improvement/nli_title_v2_selection_plan.json"


def main():
    global NAME, PLAN
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--hard-negatives", type=int, default=2)
    parser.add_argument("--frozen-output", default="frozen_selection_nli_title_v2.json")
    args = parser.parse_args()
    if args.hard_negatives < 0 or Path(args.frozen_output).name != args.frozen_output:
        parser.error("hard-negatives must be nonnegative and frozen-output must be a file name")
    NAME, PLAN = args.name, args.plan
    log = ROOT / f"artifacts/improvement/{NAME}_workflow.log"
    state = ROOT / f"artifacts/improvement/{NAME}_workflow.json"
    base = ["--base-dir", "artifacts/models/nli_large_base", "--epochs", "4", "--layers", "24",
            "--optimizer", "adamw8bit", "--learning-rate", "5e-6", "--max-length", "512",
            "--hard-negatives", str(args.hard_negatives), "--device", "cuda", "--amp", "--gradient-checkpointing",
            "--global-class-balance", "--evidence-safe-windows", "--include-title"]
    write_json(state, {"phase": "pilots", "plan_sha256": sha256_file(PLAN)})
    pilots = []
    for batch in (1, 2, 4):
        pilot_name = f"{NAME}_pilot_b{batch}"
        pilot_folder = ROOT / "artifacts/models" / pilot_name
        if not (pilot_folder / "pilot.json").exists():
            try:
                run(["scripts/train_verifier.py", "--name", pilot_name, *base, "--batch-size", str(batch),
                     "--accumulate", str(8 // batch), "--pilot-batches", str(32 // batch)], log)
            except subprocess.CalledProcessError:
                if "out of memory" not in log.read_text(encoding="utf-8")[-12000:].lower():
                    raise
                pilots.append({"batch_size": batch, "failed": "CUDA out of memory"})
                continue
        stats = read_json(pilot_folder / "pilot.json")
        pilots.append({"batch_size": batch, "accumulate": 8 // batch,
                       "seconds_per_example": stats["seconds"] / 32, **stats})
        if batch == 1:
            run(["scripts/verify_training_artifact.py", str(pilot_folder)], log)
            run(["scripts/audit_verifier_context_coverage.py", str(pilot_folder)], log)
            assert not read_json(pilot_folder / "context_coverage_audit.json")["positive_views_without_complete_annotated_rationale"]
    feasible = [p for p in pilots if "gpu_peak_reserved_gib" in p and p["gpu_peak_reserved_gib"] <= 5.2]
    if not feasible:
        raise MemoryError(f"No all-encoder pilot fits the predeclared allowance: {pilots}")
    hardware = min(feasible, key=lambda p: p["seconds_per_example"])
    write_json(ROOT / f"artifacts/improvement/{NAME}_hardware.json", {"pilots": pilots, "chosen": hardware})
    print("Selected hardware:", hardware, flush=True)
    folder = ROOT / "artifacts/models" / NAME
    if not (folder / "status.json").exists():
        write_json(state, {"phase": "training", "hardware": hardware})
        run(["scripts/train_verifier.py", "--name", NAME, *base, "--batch-size", str(hardware["batch_size"]),
             "--accumulate", str(hardware["accumulate"])], log)
    run(["scripts/verify_training_artifact.py", str(folder)], log)
    run(["scripts/audit_verifier_context_coverage.py", str(folder)], log)
    coverage = read_json(folder / "context_coverage_audit.json")
    assert not coverage["positive_views_without_complete_annotated_rationale"]
    write_json(state, {"phase": "development_scoring", "hardware": hardware})
    run(["scripts/score_verifier_checkpoints.py", "--training-run", NAME, "--epochs", "1", "2", "3", "4",
         "--include-title", "--window-overlap", "2", "--top-k", "5"], log)
    retrieval = {r["claim_id"]: {h["doc_id"]: h["score"] for h in r["hits"]}
                 for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    candidates = []
    for epoch in range(1, 5):
        path = ROOT / f"artifacts/improvement/{NAME}_epoch{epoch}_dev120/predictions.jsonl"
        rows = read_jsonl(path)
        assert len(rows) == 120
        for gate, k, support, contradict in itertools.product((None, -2, 0), (1, 3, 5),
                (0.5, 0.7, 0.9, 0.98), (0.5, 0.7, 0.9, 0.98)):
            predicted = []
            for row in rows:
                contexts = [c for c in row["contexts"] if gate is None or retrieval[row["claim_id"]][c["doc_id"]] >= gate]
                predicted.append(decide({"contexts": contexts}, view="abstract", k=k,
                                        threshold={"SUPPORT": support, "CONTRADICT": contradict}))
            metrics = verdict_metrics([row["gold"] for row in rows], predicted)
            candidates.append({"epoch": epoch, "k": k, "retrieval_gate": gate,
                               "support_threshold": support, "contradict_threshold": contradict,
                               "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                               "source_sha256": sha256_file(path)})
    candidates.sort(key=lambda c: (-c["macro_f1"], -c["accuracy"], c["epoch"], c["retrieval_gate"] is not None, c["k"]))
    frozen = {"training_name": NAME, "chosen": candidates[0], "candidates": candidates, "hardware": hardware,
              "include_title": True, "window_overlap": 2, "selection_population": "dev_tune120",
              "validation_used_for_selection": False, "plan_sha256": sha256_file(PLAN),
              "training_manifest_sha256": sha256_file(folder / "training_manifest.json")}
    frozen["selection_runner_sha256"] = sha256_file(Path(__file__))
    write_json(ROOT / "artifacts/improvement" / args.frozen_output, frozen)
    write_json(state, {"phase": "complete", "chosen": candidates[0]})
    print("Frozen development selection:", candidates[0], flush=True)


if __name__ == "__main__":
    main()
