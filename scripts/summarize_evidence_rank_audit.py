"""Compare saved ranking audit predictions without changing models or labels."""
import argparse
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.store import load_corpus, read_json, read_jsonl, sha256_file, write_json
from claim_checker.evaluation.compare import bootstrap, mcnemar_exact
from claim_checker.evaluation.metrics import (
    evidence_metrics,
    quote_validity,
    retrieval_metrics,
    verdict_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("v1", "calibrated_v2"), default="v1")
    args = parser.parse_args()
    base_path = ROOT / "artifacts/models/regularized_grouped_v1_fold0/audit_predictions.jsonl"
    name = "evidence_rank_audit_v1" if args.candidate == "v1" else "evidence_rank_calibrated_audit_v2"
    candidate_path = ROOT / f"artifacts/improvement/{name}/predictions.jsonl"
    baseline = {r["claim_id"]: r for r in read_jsonl(base_path)}
    candidate_rows = read_jsonl(candidate_path)
    candidate = {r["claim_id"]: r for r in candidate_rows}
    assert len(candidate_rows) == len(candidate) == len(baseline) == 163 and set(candidate) == set(baseline)
    ids = sorted(candidate)
    assert all(candidate[c]["gold_label"] == baseline[c]["gold_label"] and candidate[c]["claim"] == baseline[c]["claim"] for c in ids)
    gold = [baseline[c]["gold_label"] for c in ids]
    pa, pb = ([rows[c]["verdict"] if rows[c]["status"] == "ok" else "ERROR" for c in ids] for rows in (baseline, candidate))
    ma, mb = verdict_metrics(gold, pa), verdict_metrics(gold, pb)
    audit = read_json(candidate_path.parent / "metrics.json")
    assert mb == audit["verdict"] and ma == audit["baseline"]
    corpus = load_corpus(load_config().path("processed_dir"))
    only_a = sum(g == a and g != b for g, a, b in zip(gold, pa, pb, strict=True))
    only_b = sum(g == b and g != a for g, a, b in zip(gold, pa, pb, strict=True))
    quality = {}
    for name, by_id in (("baseline", baseline), ("candidate", candidate)):
        rows = [by_id[c] for c in ids]
        quality[name] = {"retrieval": retrieval_metrics(rows), "citations": evidence_metrics(rows, "cited"),
                         "quotes": quote_validity(rows, corpus)}
    report = {"population": "Same163reservedTRAINclaims; internal audit, notvalidation300", "claims": 163,
              "baseline": ma, "candidate": mb, "only_baseline_correct": only_a, "only_candidate_correct": only_b,
              "mcnemar_exact_p": mcnemar_exact(only_a, only_b), "bootstrap": bootstrap(gold, pa, pb),
              "quality": quality, "audit_gate_passed": audit["audit_gate_passed"], "requested_validation_target_assessed": False,
              "uncertainty_note": "Bootstrap resamples claims, not paper groups; internal reused TRAIN audit, not independent validation.",
              "prediction_hashes": {"baseline": sha256_file(base_path), "candidate": sha256_file(candidate_path)}}
    output = "evidence_rank_paired_audit.json" if args.candidate == "v1" else "evidence_rank_calibrated_paired_audit_v2.json"
    write_json(ROOT / "artifacts/improvement" / output, report)
    print(f"Audit163: {ma['correct']} -> {mb['correct']} correct; {ma['macro_f1']:.6f} -> {mb['macro_f1']:.6f} MacroF1; gate={audit['audit_gate_passed']}")


if __name__ == "__main__":
    main()
