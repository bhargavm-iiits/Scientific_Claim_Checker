"""Confirm exact development-selected checkpoints, with no validation threshold tuning."""
from pathlib import Path

from claim_checker.config import ExperimentsConfig, SuiteSpec, VariantSpec, load_config
from claim_checker.data.splits import normalize_claim_text
from claim_checker.data.store import load_gold, read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.analysis import analyse_run
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.runtime import ClaimChecker

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan_path = ROOT / "artifacts/improvement/selection_checkpoint_confirmation_plan.json"
    plan = read_json(plan_path)
    selection_path = ROOT / plan["settings_selection"]
    selection = read_json(selection_path)
    chosen = selection["chosen"]
    profile = ROOT / plan["profile"]
    cpu = load_config(profile)
    assert cpu.product.judge_backend == "nli" and cpu.product.evidence_mode == "evidence"
    assert cpu.verifier.model_dir == plan["reader"] and cpu.evidence_selector.model_dir == plan["selector"]
    assert cpu.retrieval.pipeline_k == chosen["k"]
    assert cpu.verifier.support_threshold == chosen["support_threshold"]
    assert cpu.verifier.contradict_threshold == chosen["contradict_threshold"]
    assert cpu.verifier.min_retrieval_score == chosen["retrieval_gate"]
    assert cpu.evidence_selector.enabled and cpu.evidence_selector.threshold == chosen["selector_threshold"]
    assert cpu.evidence_selector.maximum == chosen["max_sentences"] and cpu.evidence_selector.alpha == chosen["alpha"]
    forbidden = {normalize_claim_text(g.claim) for g in load_gold(cpu.path("processed_dir"), "validation").values()}
    contracts = {}
    for component, directory, data_file, hash_key in (
        ("reader", cpu.verifier.model_dir, "training_pairs.jsonl", "training_pairs_sha256"),
        ("selector", cpu.evidence_selector.model_dir, "training_sentences.jsonl", "training_data_sha256"),
    ):
        folder = (ROOT / directory).parent
        manifest = read_json(folder / "training_manifest.json")
        assert not manifest["settings"]["final_fit"]
        data = read_jsonl(folder / data_file)
        assert all(normalize_claim_text(r["claim"]) not in forbidden for r in data)
        assert manifest[hash_key] == sha256_file(folder / data_file)
        contracts[component] = {"manifest_sha256": sha256_file(folder / "training_manifest.json"),
                                "weights_sha256": sha256_file(ROOT / directory / "model.safetensors")}
    selector_manifest = read_json((ROOT / cpu.evidence_selector.model_dir).parent / "training_manifest.json")
    assert selector_manifest["renderer_sha256"] == sha256_file(ROOT / "src/claim_checker/models/evidence_classifier.py")
    frozen_path = ROOT / "artifacts/improvement/frozen_selection_evidence_candidate.json"
    frozen = {"plan_sha256": sha256_file(plan_path), "selection_sha256": sha256_file(selection_path),
              "profile_sha256": sha256_file(profile), "development": chosen, "contracts": contracts,
              "validation_aggregate_informed_followup_choice": True, "fresh_blind_test": False,
              "validation_examples_used_for_inference_or_training": False}
    if frozen_path.exists():
        assert read_json(frozen_path) == frozen, "Refuse to alter the frozen follow-up"
    else:
        write_json(frozen_path, frozen)
    cfg = load_config(profile, overrides={"verifier": {"device": "cuda"}, "evidence_selector": {"device": "cuda"}})
    variants = {name: VariantSpec(name=name, k=chosen["k"], judge_backend="nli", evidence_mode="evidence", oracle=oracle)
                for name, oracle in (("selection_evidence", False), ("selection_evidence_oracle", True))}
    suites = {"selection_holdout": SuiteSpec(split="accuracy_holdout", variants=["selection_evidence"]),
              "selection_recheck": SuiteSpec(split="final", variants=["selection_evidence"]),
              "selection_oracle": SuiteSpec(split="final", variants=["selection_evidence_oracle"])}
    experiments = ExperimentsConfig(variants=variants, suites=suites)
    checker = ClaimChecker(cfg, experiments=experiments)
    records, subsets = [], []
    for cohort in ("holdout", "recheck", "oracle"):
        run_id = f"selection_evidence_{cohort}_v1"
        folder = run_evaluation(cfg, experiments, checker, suite=f"selection_{cohort}", run_id=run_id)
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
    write_json(ROOT / "artifacts/improvement/final_selection_evidence_results.json", {
        "frozen_candidate_sha256": sha256_file(frozen_path), "subsets": subsets, "pooled_validation_300": pooled,
        "population_note": "Reused validation cohorts; aggregate results informed follow-up choice; not blind",
        "target_met": pooled["accuracy"] > 0.85 and pooled["macro_f1"] > 0.85,
    })
    print(f"Pooled300: accuracy={pooled['accuracy']:.6f}; Macro-F1={pooled['macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
