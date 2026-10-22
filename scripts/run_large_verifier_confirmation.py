"""Confirm the frozen larger-verifier recipe on the previously evaluated validation set."""
from pathlib import Path

from claim_checker.config import load_config, load_experiments
from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.analysis import analyse_run
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.runtime import ClaimChecker

ROOT = Path(__file__).resolve().parents[1]


def main():
    frozen_path = ROOT / "artifacts/improvement/frozen_selection_nli_large_v1.json"
    frozen = read_json(frozen_path)
    chosen = frozen["chosen"]
    cfg = load_config(ROOT / "configs/verifier_large_candidate.yaml", overrides={"verifier": {"device": "cuda"}})
    expected_path = f"artifacts/models/nli_large_scifact_final_v1/epoch{chosen['epoch']}"
    assert cfg.verifier.model_dir == expected_path
    assert cfg.verifier.support_threshold == chosen["settings"]["threshold"]["SUPPORT"]
    assert cfg.verifier.contradict_threshold == chosen["settings"]["threshold"]["CONTRADICT"]
    final_folder = ROOT / "artifacts/models/nli_large_scifact_final_v1"
    training = read_json(final_folder / "training_manifest.json")
    selection_training = read_json(ROOT / "artifacts/models" / frozen["training_name"] / "training_manifest.json")
    expected = {**selection_training["settings"], "name": final_folder.name, "epochs": chosen["epoch"], "final_fit": True,
                "schedule_epochs": selection_training["settings"]["epochs"]}
    assert training["settings"] == expected, "Refit hyperparameters must match the frozen development recipe"
    assert training["base_weights_sha256"] == selection_training["base_weights_sha256"]
    integrity = read_json(final_folder / "training_integrity.json")
    assert integrity["final_fit"] and integrity["evaluation_text_overlap"] == 0 and integrity["annotation_mismatches"] == 0
    experiments = load_experiments()
    assert experiments.variant("nli_full").k == chosen["settings"]["k"]
    checker = ClaimChecker(cfg, experiments=experiments)
    outputs, records = [], []
    suites = (("verifier_holdout", "nli_large_holdout_v1"), ("verifier_recheck", "nli_large_recheck_v1"),
              ("verifier_oracle", "nli_large_oracle_v1"))
    for suite, name in suites:
        folder = run_evaluation(cfg, experiments, checker, suite=suite, run_id=name)
        analyse_run(folder, checker.corpus)
        rows = read_jsonl(folder / "predictions.jsonl")
        metrics = verdict_metrics([r["gold_label"] for r in rows],
                                  [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
        if suite != "verifier_oracle":
            records.extend(rows)
        outputs.append({"suite": suite, "run_id": name, "metrics": metrics,
                        "target_met": metrics["accuracy"] > 0.85 and metrics["macro_f1"] > 0.85})
        print(f"{name}: accuracy={metrics['accuracy']:.4f}; Macro-F1={metrics['macro_f1']:.4f}", flush=True)
    assert len(records) == len({r["claim_id"] for r in records}) == 300
    pooled = verdict_metrics([r["gold_label"] for r in records],
                             [r["verdict"] if r["status"] == "ok" else "ERROR" for r in records])
    write_json(ROOT / "artifacts/improvement/final_large_verifier_results.json", {
        "selection_sha256": sha256_file(frozen_path), "subsets": outputs, "pooled_validation_300": pooled,
        "population_note": "Previously evaluated validation cohorts; not a new blind test",
        "target_met": pooled["accuracy"] > 0.85 and pooled["macro_f1"] > 0.85,
    })
    print(f"Pooled 300: accuracy={pooled['accuracy']:.4f}; Macro-F1={pooled['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
