"""Confirm the frozen exploratory trained-rationale pipeline on reused validation."""
from pathlib import Path

from claim_checker.config import ExperimentsConfig, SuiteSpec, VariantSpec, load_config
from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.analysis import analyse_run
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.runtime import ClaimChecker

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan_path = ROOT / "artifacts/improvement/trained_evidence_confirmation_plan.json"
    plan = read_json(plan_path)
    selection_path = ROOT / plan["selection"]
    selection = read_json(selection_path)
    chosen = selection["chosen"]
    assert {k: chosen[k] for k in plan["settings"]} == plan["settings"]
    assert selection["validation_used"] is False
    profile = ROOT / "configs/trained_evidence_candidate.yaml"
    cpu = load_config(profile)
    assert cpu.product.judge_backend == "nli" and cpu.product.evidence_mode == "evidence"
    assert cpu.verifier.model_dir == plan["reader_final"]
    assert cpu.retrieval.pipeline_k == chosen["k"]
    assert cpu.verifier.support_threshold == chosen["support_threshold"]
    assert cpu.verifier.contradict_threshold == chosen["contradict_threshold"]
    assert cpu.verifier.min_retrieval_score == chosen["retrieval_gate"]
    assert cpu.evidence_selector.enabled and cpu.evidence_selector.threshold == chosen["selector_threshold"]
    assert cpu.evidence_selector.maximum == chosen["max_sentences"] and cpu.evidence_selector.alpha == chosen["alpha"]
    selected_folder = ROOT / "artifacts/models" / plan["selector_training"]
    final_folder = ROOT / "artifacts/models" / plan["selector_final"]
    selected = read_json(selected_folder / "training_manifest.json")
    final = read_json(final_folder / "training_manifest.json")
    expected = {**selected["settings"], "name": plan["selector_final"], "epochs": plan["selector_epochs"], "final_fit": True}
    assert final["settings"] == expected, "Refit must preserve the selected training recipe and LR horizon"
    assert final["base_sha256"] == selected["base_sha256"]
    assert final["script_sha256"] == selected["script_sha256"]
    assert final["renderer_sha256"] == selected["renderer_sha256"]
    assert final["renderer_sha256"] == sha256_file(ROOT / "src/claim_checker/models/evidence_classifier.py")
    assert read_json(final_folder / "status.json")["complete"]
    integrity = read_json(final_folder / "training_integrity.json")
    assert integrity["final_fit"] and not integrity["excluded_text_overlaps"] and not integrity["source_or_annotation_mismatches"]
    assert integrity["training_data_sha256"] == sha256_file(final_folder / "training_sentences.jsonl")
    assert cpu.evidence_selector.model_dir == f"artifacts/models/{plan['selector_final']}/epoch{plan['selector_epochs']}"
    frozen_path = ROOT / "artifacts/improvement/frozen_trained_evidence_candidate.json"
    frozen = {"selection_sha256": sha256_file(selection_path), "plan_sha256": sha256_file(plan_path),
              "profile_sha256": sha256_file(profile), "development": chosen,
              "selector_training_sha256": sha256_file(final_folder / "training_manifest.json"),
              "selector_weights_sha256": sha256_file(ROOT / cpu.evidence_selector.model_dir / "model.safetensors"),
              "initial_development_gate_passed": False,
              "confirmation_exception": "Exploratory best-candidate assessment declared before validation",
              "validation_labels_used_for_tuning": False}
    if frozen_path.exists():
        assert read_json(frozen_path) == frozen, "Refuse to change the frozen candidate"
    else:
        write_json(frozen_path, frozen)
    cfg = load_config(profile, overrides={"verifier": {"device": "cuda"}, "evidence_selector": {"device": "cuda"}})
    variants = {name: VariantSpec(name=name, k=chosen["k"], judge_backend="nli", evidence_mode="evidence", oracle=oracle,
                                  description="TRAIN-only rationale classifier followed by scientific NLI judgement")
                for name, oracle in (("trained_evidence", False), ("trained_evidence_oracle", True))}
    suites = {"trained_holdout": SuiteSpec(split="accuracy_holdout", variants=["trained_evidence"]),
              "trained_recheck": SuiteSpec(split="final", variants=["trained_evidence"]),
              "trained_oracle": SuiteSpec(split="final", variants=["trained_evidence_oracle"])}
    experiments = ExperimentsConfig(variants=variants, suites=suites)
    checker = ClaimChecker(cfg, experiments=experiments)
    records, subsets = [], []
    for cohort in ("holdout", "recheck", "oracle"):
        run_id = f"trained_evidence_{cohort}_v2"
        folder = run_evaluation(cfg, experiments, checker, suite=f"trained_{cohort}", run_id=run_id)
        analyse_run(folder, checker.corpus)
        rows = read_jsonl(folder / "predictions.jsonl")
        metrics = verdict_metrics([r["gold_label"] for r in rows],
                                  [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
        errors = sum(r["status"] != "ok" for r in rows)
        subsets.append({"run_id": run_id, "metrics": metrics, "operational_errors": errors})
        if cohort != "oracle":
            records.extend(rows)
        print(f"{run_id}: accuracy={metrics['accuracy']:.6f}; Macro-F1={metrics['macro_f1']:.6f}; errors={errors}", flush=True)
    assert len(records) == len({r["claim_id"] for r in records}) == 300
    pooled = verdict_metrics([r["gold_label"] for r in records],
                             [r["verdict"] if r["status"] == "ok" else "ERROR" for r in records])
    result = {"frozen_candidate_sha256": sha256_file(frozen_path), "subsets": subsets,
              "pooled_validation_300": pooled, "population_note": "Reused validation cohorts; not a new blind test",
              "target_met": pooled["accuracy"] > 0.85 and pooled["macro_f1"] > 0.85 + 1e-12}
    write_json(ROOT / "artifacts/improvement/final_trained_evidence_results.json", result)
    print(f"Pooled300: accuracy={pooled['accuracy']:.6f}; Macro-F1={pooled['macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
