"""Summarize completed frozen-selector validation without changing predictions."""
import argparse
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.store import load_corpus, read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.compare import compare_runs, write_comparison
from claim_checker.evaluation.metrics import evidence_metrics, quote_validity, verdict_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("refit", "selected"), default="refit")
    args = parser.parse_args()
    prefix, version = ("trained_evidence", "v2") if args.family == "refit" else ("selection_evidence", "v1")
    confirmed = read_json(ROOT / f"artifacts/improvement/final_{prefix}_results.json")
    corpus = load_corpus(load_config().path("processed_dir"))
    pooled, quality = [], {}
    for cohort, expected in (("holdout", 200), ("recheck", 100), ("oracle", 100)):
        folder = ROOT / "artifacts/runs" / f"{prefix}_{cohort}_{version}"
        rows = read_jsonl(folder / "predictions.jsonl")
        assert len(rows) == len({r["claim_id"] for r in rows}) == expected
        quality[cohort] = {
            "claims": expected,
            "predictions_sha256": sha256_file(folder / "predictions.jsonl"),
            "operational_errors": sum(r["status"] != "ok" for r in rows),
            "selected": evidence_metrics(rows, "selected"),
            "cited": evidence_metrics(rows, "cited"),
            "quotes": quote_validity(rows, corpus),
        }
        if cohort != "oracle":
            pooled.extend(rows)
    assert len(pooled) == len({r["claim_id"] for r in pooled}) == 300
    metrics = verdict_metrics([r["gold_label"] for r in pooled],
                              [r["verdict"] if r["status"] == "ok" else "ERROR" for r in pooled])
    assert metrics == confirmed["pooled_validation_300"]
    quality["pooled_retrieved_300"] = {"claims": 300, "verdict": metrics,
                                       "selected": evidence_metrics(pooled, "selected"),
                                       "cited": evidence_metrics(pooled, "cited"),
                                       "quotes": quote_validity(pooled, corpus)}
    quality["population_note"] = "Reused validation cohorts; oracle is separate and never pooled."
    write_json(ROOT / f"artifacts/improvement/{prefix}_quality.json", quality)
    pairs = (("final_v2", "full_v2_nothink", f"{prefix}_vs_product"),
             ("nli_titlefamily_recheck_v1", "nli_title_full", f"{prefix}_vs_title"))
    for run, variant, name in pairs:
        comparison = compare_runs(ROOT / "artifacts/runs", (run, variant),
                                  (f"{prefix}_recheck_{version}", prefix))
        assert comparison["n_claims"] == 100
        write_comparison(ROOT, comparison, name)
    print("Saved citation quality and two paired historical100 comparisons.")


if __name__ == "__main__":
    main()
