"""Compare the fixed candidate and baseline on exactly the same300claimIDs."""
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.store import load_corpus, read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.compare import bootstrap, mcnemar_exact
from claim_checker.evaluation.metrics import evidence_metrics, quote_validity, verdict_metrics

ROOT = Path(__file__).resolve().parents[1]


def load(prefix):
    folders = [ROOT / "artifacts/runs" / f"{prefix}_{cohort}_v1" for cohort in ("recheck", "holdout")]
    rows = [r for folder in folders for r in read_jsonl(folder / "predictions.jsonl")]
    assert len(rows) == len({r["claim_id"] for r in rows}) == 300
    return {r["claim_id"]: r for r in rows}, {folder.name: sha256_file(folder / "predictions.jsonl") for folder in folders}


def main():
    baseline, a_hashes = load("nli_titlefamily")
    candidate, b_hashes = load("roberta_scifact")
    assert set(baseline) == set(candidate)
    ids = sorted(baseline)
    assert all(baseline[c]["gold_label"] == candidate[c]["gold_label"] and baseline[c]["claim"] == candidate[c]["claim"] for c in ids)
    gold = [baseline[c]["gold_label"] for c in ids]
    pa, pb = ([rows[c]["verdict"] if rows[c]["status"] == "ok" else "ERROR" for c in ids] for rows in (baseline, candidate))
    ma, mb = verdict_metrics(gold, pa), verdict_metrics(gold, pb)
    assert mb == read_json(ROOT / "artifacts/improvement/final_roberta_verifier_results.json")["pooled_validation_300"]
    only_a = sum(a == g and b != g for a, b, g in zip(pa, pb, gold, strict=True))
    only_b = sum(b == g and a != g for a, b, g in zip(pa, pb, gold, strict=True))
    rows = [candidate[c] for c in ids]
    corpus = load_corpus(load_config().path("processed_dir"))
    report = {"claims": 300, "population": "Unchanged reused SciFactvalidation300; pairedcomparison,notblind",
              "baseline": ma, "candidate": mb, "only_baseline_correct": only_a, "only_candidate_correct": only_b,
              "mcnemar_exact_p": mcnemar_exact(only_a, only_b), "bootstrap": bootstrap(gold, pa, pb),
              "operational_errors": sum(r["status"] != "ok" for r in rows), "citations": evidence_metrics(rows, "cited"),
              "quotes": quote_validity(rows, corpus), "prediction_hashes": {**a_hashes, **b_hashes},
              "target_met": mb["accuracy"] > 0.95 and mb["macro_f1"] > 0.95}
    write_json(ROOT / "artifacts/improvement/roberta_paired_quality.json", report)
    print(f"Paired300: baseline{ma['correct']}correct, candidate{mb['correct']}correct; allquotes/sourcechecksreported")


if __name__ == "__main__":
    main()
