"""Freeze one new reader recipe without searching the reused validation labels."""
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import normalize_claim_text
from claim_checker.data.store import load_gold, read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "artifacts/improvement/scifact95_roberta_protocol_v1.json"


def main():
    cfg = load_config()
    old_path = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    old = read_json(old_path)
    train = load_gold(cfg.path("processed_dir"), "train")
    validation = load_gold(cfg.path("processed_dir"), "validation")
    stop = old["folds"][0]["stop"]
    fitting = sorted(set(old["group_by_claim"]) - set(stop))
    assert len(fitting) == 647 and len(stop) == 160
    papers = {name: {d for cid in ids for d in train[cid].reference_doc_ids} for name, ids in (("fit", fitting), ("stop", stop))}
    assert not papers["fit"] & papers["stop"]
    forbidden = {normalize_claim_text(g.claim) for g in validation.values()}
    assert all(normalize_claim_text(train[cid].claim) not in forbidden for cid in fitting + stop)
    protocol = {
        "objective": "Same SciFact validation300: accuracy>0.95 and three-class Macro-F1>0.95",
        "baseline": {"correct": 242, "claims": 300, "accuracy": 242 / 300, "macro_f1": 0.7959743387519879},
        "minimum_correct_for_strict_accuracy_target": 286,
        "architecture": "Independent generic RoBERTa-large NLI base; no SciFact pretrained checkpoint",
        "base_dir": "artifacts/models/roberta_large_nli_base",
        "source_grouped_protocol_sha256": sha256_file(old_path),
        "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
        "validation_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
        "fit_ids": fitting, "stop_ids": stop, "eligible_ids": sorted(fitting + stop),
        "selection": "Minimum unweighted annotated stop-pair cross-entropy, maximum4epochs,patience1,min_delta0.002",
        "recipe": {"seed": 42, "maximum_epochs": 4, "patience": 1, "minimum_loss_improvement": 0.002,
                   "learning_rate": 1e-5, "weight_decay": 0.05, "dropout": 0.2,
                   "physical_batch": 2, "effective_batch": 16, "hard_negatives": 1,
                   "trainable": "All24encoderlayers and classification head; freeze token embeddings",
                   "max_length": 512, "class_weighting": "Global inverse-frequency weights",
                   "optimizer": "AdamW8bit", "precision": "FP32weights,BF16CUDAautocast", "gradient_checkpointing": True},
        "inference": {"k": 3, "gate": -2, "support_threshold": 0.5, "contradict_threshold": 0.5,
                      "include_title": True, "window_overlap": 2},
        "refit": "After epoch selection, refit once from generic base on eligible807TRAINclaims for chosenepochs using original4epochLRhorizon",
        "selection_budget": {"recipes": 1, "threshold_searches": 0, "validation_confirmations": 1},
        "population_note": "Validation300 is repeatedly reused, not blind. TRAIN stop groups are not an independent test.",
        "reporting": "Count all300claims, all3classes and operational failures; preserve baseline and do not claim success if either target is unmet",
        "scope": "This new architecture has its own single grouped stopping split; it does not complete or replace the older5foldDeBERTaprotocol",
    }
    if PATH.exists():
        assert read_json(PATH) == protocol, "Preserve the frozen recipe; use a new protocol for changes"
    else:
        write_json(PATH, protocol)
    print("Frozen647fit/160paper-disjointstop;807finalTRAIN;300validationunchanged; target>=286correctandMacroF1>0.95")


if __name__ == "__main__":
    main()
