"""Audit saved curves, model selection and cohort composition without retuning."""
from collections import Counter
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.splits import load_split, normalize_claim_text
from claim_checker.data.store import load_gold, read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/improvement/overfitting_audit.json"


def main():
    cfg = load_config()
    train, validation = load_gold(cfg.path("processed_dir"), "train"), load_gold(cfg.path("processed_dir"), "validation")
    excluded = {normalize_claim_text(g.claim) for g in validation.values()}
    dev_ids = set(load_split(cfg, "dev_tune")["claim_ids"])
    smoke_ids = set(load_split(cfg, "smoke")["claim_ids"])
    tuning_texts = {normalize_claim_text(train[c].claim) for c in dev_ids | smoke_ids}
    fitted = [g for g in train.values() if not g.conflict
              and normalize_claim_text(g.claim) not in excluded | tuning_texts]
    fitted_docs = {d for g in fitted for d in g.reference_doc_ids}

    def composition(items):
        items = list(items)
        docs = {d for g in items for d in g.reference_doc_ids}
        return {"claims": len(items), "labels": dict(Counter(g.label for g in items)),
                "unique_reference_docs": len(docs), "reference_docs_seen_in_selection_train": len(docs & fitted_docs),
                "claims_with_any_reference_doc_seen": sum(bool(set(g.reference_doc_ids) & fitted_docs) for g in items),
                "claims_with_all_reference_docs_seen": sum(bool(g.reference_doc_ids) and set(g.reference_doc_ids) <= fitted_docs for g in items)}

    selector_folder = ROOT / "artifacts/models/evidence_deberta_train_v2"
    curves = sorted(read_json(selector_folder / "selection.json")["checkpoints"], key=lambda r: r["epoch"])
    nli_folder = ROOT / "artifacts/models/nli_large_title_train_v2"
    nli_selection = read_json(ROOT / "artifacts/improvement/frozen_selection_nli_title_family_v1.json")
    nli_curves = []
    fixed_keys = ("k", "retrieval_gate", "support_threshold", "contradict_threshold")
    fixed_settings = {key: nli_selection["chosen"][key] for key in fixed_keys}
    for epoch in range(1, 5):
        candidates = [c for c in nli_selection["candidates"] if c["epoch"] == epoch]
        winner = max(candidates, key=lambda c: (c["macro_f1"], c["accuracy"]))
        fixed = next(c for c in candidates if all(c[key] == fixed_settings[key] for key in fixed_keys))
        loss = read_json(nli_folder / f"epoch{epoch}/training_loss.json")
        nli_curves.append({"epoch": epoch, "mean_training_loss": loss["mean_loss"],
                           "best_development_accuracy": winner["accuracy"], "best_development_macro_f1": winner["macro_f1"],
                           "fixed_setting_accuracy": fixed["accuracy"], "fixed_setting_macro_f1": fixed["macro_f1"],
                           "note": "Each epoch's best settings are optimistic; not a fixed-threshold learning curve"})
    searches = []
    for name in ("selection", "reader_calibration_refinement", "selector_calibration_refinement"):
        path = ROOT / f"artifacts/improvement/trained_evidence_deberta_dev120_v2/{name}.json"
        data = read_json(path)
        count = data.get("tested_candidates", len(data.get("candidates", [])))
        searches.append({"artifact": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                         "tested_settings": count, "chosen": data["chosen"]})
    cohorts = {}
    for family, suffix in (("trained_evidence", "v2"), ("selection_evidence", "v1")):
        result = read_json(ROOT / f"artifacts/improvement/final_{family}_results.json")
        cohorts[family] = {"pooled_accuracy": result["pooled_validation_300"]["accuracy"],
                           "pooled_macro_f1": result["pooled_validation_300"]["macro_f1"],
                           "operational_errors": sum(s["operational_errors"] for s in result["subsets"]),
                           "prediction_files": {cohort: sha256_file(ROOT / f"artifacts/runs/{family}_{cohort}_{suffix}/predictions.jsonl")
                                                for cohort in ("holdout", "recheck", "oracle")}}
    source_audits = {"selector": read_json(selector_folder / "training_integrity.json"),
                     "reader": read_json(nli_folder / "training_integrity.json")}
    # Saved data audit: exact normalized-text overlap is distinct from same-paper dependence.
    fitted_texts = {normalize_claim_text(g.claim) for g in fitted}
    assert not fitted_texts & excluded
    assert not fitted_texts & {normalize_claim_text(train[c].claim) for c in dev_ids}
    audit = {"purpose": "Post-experiment diagnosis, not model selection or a new blind score",
             "selector_learning_curve": curves, "nli_learning_curve": nli_curves, "nli_fixed_settings": fixed_settings,
             "selection_training_claims": len(fitted),
             "development_composition": composition(train[c] for c in dev_ids),
             "validation_composition": composition(validation.values()),
             "development_searches": searches,
             "settings_tested_in_these_three_searches": sum(s["tested_settings"] for s in searches),
             "search_count_note": "Settings are correlated, not independent experiments; other past searches are additional.",
             "cohorts": cohorts, "source_audits": source_audits,
             "normalized_selection_train_development_overlap": 0,
             "normalized_selection_train_validation_overlap": 0,
             "fresh_blind_validation": False}
    write_json(OUT, audit)
    print({k: audit[k] for k in ("selection_training_claims", "development_composition", "validation_composition",
                                "settings_tested_in_these_three_searches", "normalized_selection_train_validation_overlap")})
    print("Saved", OUT)


if __name__ == "__main__":
    main()
