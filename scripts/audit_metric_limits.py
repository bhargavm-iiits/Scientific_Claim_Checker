"""Aggregate mathematical metric ceilings; never export evaluation training cases."""
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.store import load_gold, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = load_config()
    gold = list(load_gold(cfg.path("processed_dir"), "validation").values())
    assert len(gold) == 300 and all(g.reference_doc_ids for g in gold)
    total = sum(len(ids) for g in gold for ids in g.evidence_by_doc.values())
    report = {"population": "Original unchanged validation300; aggregate structural audit only",
              "claims": len(gold), "gold_evidence_sentences": total,
              "reference_document_recall_upper_bounds": {
                  f"recall@{k}": sum(min(k, len(g.reference_doc_ids)) / len(g.reference_doc_ids) for g in gold) / len(gold)
                  for k in (1, 3, 5, 10)},
              "two_quotes_per_document_evidence_recall_upper_bound_unlimited_documents":
                  sum(min(2, len(ids)) for g in gold for ids in g.evidence_by_doc.values()) / total,
              "evidence_recall_upper_bounds_by_document_budget_without_quote_cap": {
                  str(k): sum(sum(sorted((len(ids) for ids in g.evidence_by_doc.values()), reverse=True)[:k]) for g in gold) / total
                  for k in (1, 3, 5, 10)},
              "two_quotes_per_document_and_three_documents_evidence_recall_upper_bound":
                  sum(sum(sorted((min(2, len(ids)) for ids in g.evidence_by_doc.values()), reverse=True)[:3]) for g in gold) / total,
              "gold_sha256": sha256_file(cfg.path("processed_dir") / "claim_gold_validation.jsonl"),
              "limitations": "Upper bounds assume perfect selection of annotated sources/sentences. These are not measured model scores. No claims, annotations or examples are exported to training or inference. Removing a structural cap does not establish achievable95percent quality."}
    write_json(ROOT / "artifacts/improvement/metric_limits_20261023.json", report)
    print(report)


if __name__ == "__main__":
    main()
