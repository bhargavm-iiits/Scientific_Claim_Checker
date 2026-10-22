"""Freeze one ranking experiment, with an application audit before validation."""
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.store import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "artifacts/improvement/evidence_rank_protocol_v1.json"


def main():
    cfg = load_config()
    original = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    grouped = read_json(original)
    split = grouped["folds"][0]
    reader = ROOT / "artifacts/models/regularized_grouped_v1_fold0"
    selected = read_json(reader / "selection.json")
    final_reader = ROOT / "artifacts/models/nli_large_titlefamily_final_v1/epoch1"
    plan = {
        "purpose": "Learn source-evidence ranking; no validation-label fitting or threshold search",
        "target": "Same300SciFactclaims: accuracy>0.95,Macro-F1>0.95; no guarantee",
        "grouped_protocol_sha256": sha256_file(original), "fit_ids": split["train"], "stop_ids": split["stop"],
        "audit_ids": split["audit"], "eligible_ids": sorted(grouped["group_by_claim"]),
        "train_gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_train.jsonl"),
        "base": "cross-encoder/ms-marco-MiniLM-L6-v2", "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "recipe": {"seed": 42, "max_length": 512, "maximum_epochs": 4, "patience": 1, "min_delta": 0.001,
                   "learning_rate": 5e-6, "dropout": 0.2, "weight_decay": 0.05, "physical_batch": 8,
                   "effective_batch": 32, "hard_negative_documents": 4, "loss": "Unweighted binary cross-entropy",
                   "trainable": "All6encoderlayers/classifier; embeddings frozen",
                   "selection": "Mean stop-window BCE on annotated and explicitly weak labels; no application threshold grid"},
        "candidates": "Original BM25+dense RRF top20; no gold IDs used to construct retrieval candidates",
        "training": "Reference documents plus up to4nonreference RRF negatives. Complete annotated rationales give1; NEI/weak negatives give0. Omit partial rationale and oversized training windows, record omissions; evaluation keeps allclaims.",
        "separation": "Fit/stop claim groups disjoint; protect stop/audit reference papers from fitting; stop source docs exclude all fitted source docs and audit reference papers",
        "audit_reader": str((reader / f"epoch{selected['epoch']}").relative_to(ROOT)),
        "audit_reader_sha256": selected["weights_sha256"], "audit_baseline_path": str((reader / "audit_metrics.json").relative_to(ROOT)),
        "audit_baseline_sha256": sha256_file(reader / "audit_metrics.json"),
        "audit_gate": "All163realapplicationchecks: accuracy AND Macro-F1 improve by >=0.02 over frozen same-reader baseline, zero operational errors. Otherwise reject without validation300.",
        "final_reader": str(final_reader.relative_to(ROOT)), "final_reader_sha256": sha256_file(final_reader / "model.safetensors"),
        "reader_transfer": "Audit uses fixed484TRAINreader to keep audit genuinely unfitted; final validation pairs the learned ranker with the existing stronger807TRAINreader. Audit score is not an estimate for this final combination.",
        "inference": {"k": 3, "rerank_top_n": 20, "gate": 0, "support_threshold": 0.5,
                      "contradict_threshold": 0.9, "include_title": True, "window_overlap": 2},
        "refit": "Only if auditgatepasses, refit ranker from genericbase on807TRAIN for chosenepochs with original4epochLRhorizon; keep finalreaderfixed",
        "budget": {"recipes": 1, "threshold_grids": 0, "validation_confirmations": 1},
        "limitations": "TRAIN audit used as accept/reject development evidence, not blind. Validation300reused. Nonreference/nonrationale negatives may be incomplete annotations; neither relevance nor labels are certified scientific truth.",
    }
    if PATH.exists():
        assert read_json(PATH) == plan, "Use a new version for a changed experiment"
    else:
        write_json(PATH, plan)
    print(f"Frozen{len(split['train'])}fit/{len(split['stop'])}stop/{len(split['audit'])}audit; validation gated before use")


if __name__ == "__main__":
    main()
