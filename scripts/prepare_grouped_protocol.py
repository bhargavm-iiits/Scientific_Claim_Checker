"""Freeze an anti-overfitting TRAIN-only protocol; preserve historical split files."""
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.grouped_training import grouped_folds
from claim_checker.data.store import load_gold, read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = load_config()
    protocol = grouped_folds(load_gold(cfg.path("processed_dir"), "train"),
                             load_gold(cfg.path("processed_dir"), "validation"))
    protocol.update({"train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
                     "split_code_sha256": sha256_file(ROOT / "src/claim_checker/data/grouped_training.py"),
                     "purpose": "TRAIN-only grouped repeated holdouts; not a new external blind test",
                     "audit_use": "Only after epoch selection and settings are frozen; never tune on audit predictions",
                     "recipe": {"base_dir": "artifacts/models/nli_large_base", "layers": 4,
                                "maximum_epochs": 4, "patience": 1, "minimum_loss_improvement": 0.001,
                                "learning_rate": 5e-6, "weight_decay": 0.05, "dropout": 0.2,
                                "physical_batch": 2, "effective_batch": 16, "hard_negatives": 1,
                                "selection": "Mean unweighted annotated stop-pair cross-entropy; no threshold grid",
                                "k": 3, "retrieval_gate": -2, "support_threshold": 0.5, "contradict_threshold": 0.9},
                     "experiment_budget": {"training_recipes": 1, "threshold_searches": 0,
                                           "final_refit": "Disabled until all grouped folds have been assessed"}})
    path = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    if path.exists():
        assert read_json(path) == protocol, "Use a new protocol name if the recipe changes"
    else:
        write_json(path, protocol)
    print({"eligible_claims": protocol["eligible_claims"], "independent_groups": protocol["independent_groups"],
           "sizes": [{key: len(fold[key]) for key in ("train", "stop", "audit")} for fold in protocol["folds"]]})


if __name__ == "__main__":
    main()
