"""Replay saved decoder outputs to verify workflow parity and source quotations.

This is an integration check, not a second model evaluation or fresh accuracy score.
No decoder is loaded and no model calls are made.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from claim_checker.config import load_config
from claim_checker.data.store import (
    load_corpus,
    load_gold,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from claim_checker.evaluation.metrics import evidence_metrics, quote_validity
from claim_checker.evaluation.runner import prediction_record
from claim_checker.retrieval.index import RetrievalOutput
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import RetrievalHit

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    if not read_json(args.folder / "status.json")["complete"]:
        raise ValueError("Complete saved decoder predictions are required")
    rows = read_jsonl(args.folder / "predictions.jsonl")
    cfg = load_config(ROOT / "configs/lora_candidate.yaml")
    corpus = load_corpus(cfg.path("processed_dir"))
    gold = load_gold(cfg.path("processed_dir"), "train")
    by_text = {r["claim"]: r for r in rows}
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}

    class ReplayRetriever:
        fingerprint = "saved-retrieval-replay"

        def retrieve(self, claim, **kwargs):
            hits = retrieval[by_text[claim]["claim_id"]]
            return RetrievalOutput(hits=[RetrievalHit.model_validate(h) for h in hits], timings={})

    class ReplayJudge:
        def judge(self, claim, doc):
            row = by_text[claim]
            for result in row["documents"]:
                if result["doc_id"] == doc.doc_id:
                    return result["windows"]
            raise ValueError(row["error"] or "Saved decoder document is missing")

    checker = ClaimChecker(cfg, corpus=corpus, retriever=ReplayRetriever(), check_resources=False)
    checker._lora_judge = ReplayJudge()
    records = []
    for row in rows:
        result = checker.check(row["claim"], claim_id=row["claim_id"])
        expected = row["verdict"] if row["error"] is None else None
        if result.verdict != expected or (result.status == "ok") != (row["error"] is None):
            raise AssertionError(f"Component/workflow prediction mismatch for {row['claim_id']}")
        if row["error"] is None and result.conflict != row["conflict"]:
            raise AssertionError("Window/document conflict was lost by workflow integration")
        records.append(prediction_record("replay_only", gold[row["claim_id"]], result))
    report = {"purpose": "Saved-output integration parity, not new model evaluation", "claims": len(rows),
              "label_or_status_mismatches": 0, "predictions_sha256": sha256_file(args.folder / "predictions.jsonl"),
              "citation_metrics_on_saved_outputs": evidence_metrics(records, "cited"),
              "source_quote_validity": quote_validity(records, corpus)}
    write_json(args.folder / "pipeline_replay.json", report)
    print(report)


if __name__ == "__main__":
    main()
