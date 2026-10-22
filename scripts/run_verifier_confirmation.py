"""Run the frozen scientific verifier on all 300 public validation claims.

The new 200-claim confirmation population is scored first. The earlier final-100
is a historical recheck. There is no parameter selection in this script.
"""
from pathlib import Path

import torch

from claim_checker.config import load_config, load_experiments
from claim_checker.data.store import read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.analysis import analyse_run
from claim_checker.evaluation.metrics import verdict_metrics
from claim_checker.evaluation.runner import run_evaluation
from claim_checker.runtime import ClaimChecker

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    cfg = load_config(ROOT / "tmp/verifier_eval_gpu.yaml")
    frozen_path = ROOT / "artifacts/improvement/frozen_selection_v2.json"
    frozen = read_json(frozen_path)
    assert cfg.verifier.support_threshold == frozen["support_threshold"]
    assert cfg.verifier.contradict_threshold == frozen["contradict_threshold"]
    assert cfg.verifier.model_dir == f"artifacts/models/{frozen['final_fit']['name']}/epoch{frozen['final_fit']['epochs']}"
    integrity = read_json(ROOT / "artifacts/models" / frozen["final_fit"]["name"] / "training_integrity.json")
    assert integrity["final_fit"] and integrity["evaluation_text_overlap"] == 0
    experiments = load_experiments()
    assert experiments.variant("nli_full").k == frozen["k"]
    checker = ClaimChecker(cfg, experiments=experiments)
    outputs = []
    records = []
    for suite, name in (("verifier_holdout", "verifier_holdout_v2"),
                        ("verifier_recheck", "verifier_recheck_v2")):
        folder = run_evaluation(cfg, experiments, checker, suite=suite, run_id=name)
        analyse_run(folder, checker.corpus)
        rows = read_jsonl(folder / "predictions.jsonl")
        records.extend(rows)
        metrics = verdict_metrics([r["gold_label"] for r in rows],
                                  [r["verdict"] if r["status"] == "ok" else "ERROR" for r in rows])
        outputs.append({"suite": suite, "run_id": name, "metrics": metrics,
                        "target_met": metrics["accuracy"] > 0.85 and metrics["macro_f1"] > 0.85})
        print(f"{name}: {metrics['accuracy']:.4f} accuracy, {metrics['macro_f1']:.4f} Macro-F1", flush=True)
    assert len(records) == len({r["claim_id"] for r in records}) == 300
    pooled = verdict_metrics([r["gold_label"] for r in records],
                             [r["verdict"] if r["status"] == "ok" else "ERROR" for r in records])
    write_json(ROOT / "artifacts/improvement/final_verifier_results.json", {
        "selection_sha256": sha256_file(frozen_path), "subsets": outputs,
        "pooled_validation_300": pooled,
        "target_met": pooled["accuracy"] > 0.85 and pooled["macro_f1"] > 0.85,
    })
    print(f"Pooled 300: {pooled['accuracy']:.4f} accuracy, {pooled['macro_f1']:.4f} Macro-F1", flush=True)


if __name__ == "__main__":
    main()
