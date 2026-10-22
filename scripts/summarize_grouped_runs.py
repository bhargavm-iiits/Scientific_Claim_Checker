"""Report completed grouped TRAIN audits, never substitute them for validation300."""
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.store import load_corpus, read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.analysis import variant_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "artifacts/improvement/grouped_training_protocol_v1.json"
    protocol = read_json(path)
    corpus = load_corpus(load_config().path("processed_dir"))
    records, completed = [], []
    for split in protocol["folds"]:
        folder = ROOT / f"artifacts/models/regularized_grouped_v1_fold{split['fold']}"
        if not (folder / "audit_metrics.json").exists():
            continue
        assert read_json(folder / "identity.json")["protocol_sha256"] == sha256_file(path)
        rows = read_jsonl(folder / "audit_predictions.jsonl")
        assert len(rows) == len({r["claim_id"] for r in rows}) == len(split["audit"])
        assert {r["claim_id"] for r in rows} == set(split["audit"])
        selected = read_json(folder / "selection.json")
        curve = read_jsonl(folder / "learning_curve.jsonl")
        completed.append({"fold": split["fold"], "chosen_epoch": selected["epoch"], "epochs_trained": len(curve),
                          "audit_claims": len(rows), "verdict": variant_metrics(rows, corpus, None)["verdict"],
                          "predictions_sha256": sha256_file(folder / "audit_predictions.jsonl")})
        records.extend(rows)
    if not completed:
        raise ValueError("No grouped audit fold has completed")
    assert len(records) == len({r["claim_id"] for r in records})
    complete = len(completed) == protocol["n_folds"]
    if complete:
        assert len(records) == protocol["eligible_claims"]
    summary = {"population": "Grouped held-out original TRAIN claims; not external validation300",
               "requested_validation_target_assessed": False, "full_grouped_assessment_complete": complete,
               "folds": completed, "pooled_completed_train_audits": variant_metrics(records, corpus, None),
               "final_refit_allowed": complete, "protocol_sha256": sha256_file(path)}
    write_json(ROOT / "artifacts/improvement/regularized_grouped_summary.json", summary)
    verdict = summary["pooled_completed_train_audits"]["verdict"]
    print(f"Grouped TRAIN audits: folds={len(completed)}/{protocol['n_folds']}; claims={len(records)}; "
          f"accuracy={verdict['accuracy']:.6f}; Macro-F1={verdict['macro_f1']:.6f}; not validation300.")


if __name__ == "__main__":
    main()
