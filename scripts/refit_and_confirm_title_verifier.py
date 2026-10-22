"""Refit the frozen title-aware development recipe, then confirm on reused validation."""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from claim_checker.config import ExperimentsConfig, SuiteSpec, VariantSpec, load_config
from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.analysis import analyse_run
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.runtime import ClaimChecker

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/improvement/frozen_selection_nli_title_v2.json"
FINAL_NAME = "nli_large_title_final_v2"
PROFILE = ROOT / "configs/verifier_title_candidate.yaml"


def run(arguments):
    print("Starting:", " ".join(arguments), flush=True)
    subprocess.run([sys.executable, "-X", "utf8", *arguments], cwd=ROOT, check=True)


def experiments_for(k):
    variants = {
        name: VariantSpec(name=name, k=k, judge_backend="nli", evidence_mode="full_abstract", oracle=oracle)
        for name, oracle in (("nli_title_full", False), ("nli_title_oracle", True))
    }
    suites = {
        "title_holdout": SuiteSpec(split="accuracy_holdout", variants=["nli_title_full"]),
        "title_recheck": SuiteSpec(split="final", variants=["nli_title_full"]),
        "title_oracle": SuiteSpec(split="final", variants=["nli_title_oracle"]),
    }
    return ExperimentsConfig(variants=variants, suites=suites)


def main():
    global FROZEN, FINAL_NAME, PROFILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--final-name", default=FINAL_NAME)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--run-prefix", default="nli_title")
    parser.add_argument("--run-version", default="v2")
    parser.add_argument("--plan-file", type=Path, default=ROOT / "artifacts/improvement/nli_title_v2_refit_plan.json")
    parser.add_argument("--result-file", type=Path, default=ROOT / "artifacts/improvement/final_title_verifier_results.json")
    args = parser.parse_args()
    FROZEN, FINAL_NAME, PROFILE = args.frozen, args.final_name, args.profile
    frozen = read_json(FROZEN)
    assert frozen["validation_used_for_selection"] is False
    chosen = frozen["chosen"]
    if "family_candidates" in frozen:
        folders = []
        for candidate in frozen["family_candidates"]:
            record = read_json(ROOT / candidate["path"])
            folders.append(ROOT / f"artifacts/improvement/{record['training_name']}_epoch{record['chosen']['epoch']}_dev120")
        run(["scripts/analyze_verifier_ensemble.py", "--folders", *map(str, folders), "--ks", "1", "3", "5",
             "--name", "nli_title_ensemble_diagnostic"])
        diagnostic = read_json(ROOT / "artifacts/improvement/nli_title_ensemble_diagnostic.json")
        best_mixture = next(c for c in diagnostic["candidates"] if 0 < c["second_weight"] < 1)
        qualifies = (best_mixture["accuracy"] > 0.85 and best_mixture["macro_f1"] > 0.85
                     and best_mixture["macro_f1"] >= chosen["macro_f1"] + 0.02)
        write_json(ROOT / "artifacts/improvement/nli_title_ensemble_decision.json", {
            "plan_sha256": sha256_file(ROOT / "artifacts/improvement/nli_title_ensemble_diagnostic_plan.json"),
            "population": "dev_tune120", "validation_used": False, "single_model": chosen,
            "best_mixture": best_mixture, "implementation_criterion_passed": qualifies,
        })
        if qualifies:
            raise RuntimeError("The predeclared ensemble criterion passed; implement and freeze it before any title-family validation")
    run(["scripts/verify_nli_pipeline_replay.py", "--frozen", str(FROZEN)])
    training_folder = ROOT / "artifacts/models" / frozen["training_name"]
    selected = read_json(training_folder / "training_manifest.json")
    assert sha256_file(training_folder / "training_manifest.json") == frozen["training_manifest_sha256"]
    expected = {**selected["settings"], "name": FINAL_NAME, "epochs": chosen["epoch"], "final_fit": True,
                "schedule_epochs": selected["settings"]["epochs"]}
    final_folder = ROOT / "artifacts/models" / FINAL_NAME
    plan = {"selection_sha256": sha256_file(FROZEN), "settings": expected,
            "inference": chosen, "validation_policy": "No validation labels used for selection or refit"}
    plan_path = args.plan_file
    if plan_path.exists():
        assert read_json(plan_path) == plan, "Refuse to change a frozen refit plan"
    else:
        write_json(plan_path, plan)
    if not (final_folder / "status.json").exists():
        arguments = ["scripts/train_verifier.py"]
        for key in ("name", "base_dir", "epochs", "schedule_epochs", "layers", "batch_size", "accumulate",
                    "learning_rate", "max_length", "hard_negatives", "device", "threads", "optimizer"):
            arguments.extend(["--" + key.replace("_", "-"), str(expected[key])])
        for key in ("amp", "gradient_checkpointing", "global_class_balance", "final_fit", "evidence_safe_windows", "include_title"):
            if expected[key]:
                arguments.append("--" + key.replace("_", "-"))
        run(arguments)
    actual = read_json(final_folder / "training_manifest.json")
    assert actual["settings"] == expected
    assert actual["base_weights_sha256"] == selected["base_weights_sha256"]
    assert actual["script_sha256"] == selected["script_sha256"], "Training implementation changed after selection"
    assert read_json(final_folder / "status.json")["complete"] is True
    run(["scripts/verify_training_artifact.py", str(final_folder)])
    run(["scripts/audit_verifier_context_coverage.py", str(final_folder)])
    integrity = read_json(final_folder / "training_integrity.json")
    assert integrity["evaluation_text_overlap"] == integrity["annotation_mismatches"] == 0
    assert integrity["final_fit"] is True
    coverage = read_json(final_folder / "context_coverage_audit.json")
    assert not coverage["positive_views_without_complete_annotated_rationale"]
    profile = {
        "product": {"prompt_set": "v1", "judge_think": False, "evidence_mode": "full_abstract", "judge_backend": "nli"},
        "retrieval": {"pipeline_k": chosen["k"]},
        "verifier": {"model_dir": f"artifacts/models/{FINAL_NAME}/epoch{chosen['epoch']}", "device": "cpu",
                     "include_title": frozen["include_title"], "window_overlap": frozen["window_overlap"],
                     "support_threshold": chosen["support_threshold"], "contradict_threshold": chosen["contradict_threshold"],
                     "min_retrieval_score": chosen["retrieval_gate"]},
    }
    encoded = "# Experimental title-aware NLI alternative; see the separate validation report.\n" + yaml.safe_dump(profile, sort_keys=False)
    if PROFILE.exists():
        assert PROFILE.read_text(encoding="utf-8") == encoded, "Refuse to replace a different candidate profile"
    else:
        PROFILE.write_text(encoded, encoding="utf-8")
    cfg = load_config(PROFILE, overrides={"verifier": {"device": "cuda"}})
    experiments = experiments_for(chosen["k"])
    checker = ClaimChecker(cfg, experiments=experiments)
    summaries, retrieved = [], []
    for cohort in ("holdout", "recheck", "oracle"):
        suite, run_id = f"title_{cohort}", f"{args.run_prefix}_{cohort}_{args.run_version}"
        folder = run_evaluation(cfg, experiments, checker, suite=suite, run_id=run_id)
        analyse_run(folder, checker.corpus)
        rows = read_jsonl(folder / "predictions.jsonl")
        metrics = verdict_metrics([r["gold_label"] for r in rows],
                                  [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
        errors = sum(r["status"] != "ok" for r in rows)
        summaries.append({"suite": suite, "run_id": run_id, "metrics": metrics, "operational_errors": errors,
                          "target_met": metrics["accuracy"] > 0.85 and metrics["macro_f1"] > 0.85})
        if suite != "title_oracle":
            retrieved.extend(rows)
        print(f"{run_id}: accuracy={metrics['accuracy']:.4f}; Macro-F1={metrics['macro_f1']:.4f}; errors={errors}", flush=True)
    assert len(retrieved) == len({r["claim_id"] for r in retrieved}) == 300
    pooled = verdict_metrics([r["gold_label"] for r in retrieved],
                             [r["verdict"] if r["status"] == "ok" else "ERROR" for r in retrieved])
    result = {"selection_sha256": sha256_file(FROZEN), "config_sha256": sha256_file(PROFILE),
              "subsets": summaries, "pooled_validation_300": pooled,
              "population_note": "Previously evaluated validation cohorts; not a new blind test",
              "target_met": pooled["accuracy"] > 0.85 and pooled["macro_f1"] > 0.85}
    write_json(args.result_file, result)
    print(f"Pooled 300: accuracy={pooled['accuracy']:.4f}; Macro-F1={pooled['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
