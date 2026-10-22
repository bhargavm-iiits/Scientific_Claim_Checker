"""Check CPU/GPU label parity on fixed saved retrieval inputs, without gold inputs."""
import argparse
from pathlib import Path

import torch

from claim_checker.config import load_config
from claim_checker.data.store import load_corpus, read_jsonl, write_json
from claim_checker.retrieval.index import RetrievalOutput
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import RetrievalHit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/verifier_candidate.yaml")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/runs/verifier_recheck_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/improvement/cpu_parity.json")
    parser.add_argument("--size", type=int, default=5)
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size must be positive")
    cfg = load_config(args.config, overrides={"verifier": {"device": "cpu"}})
    if cfg.product.judge_backend != "nli":
        raise ValueError("This parity check is for the NLI verifier backend")
    folder = args.run_dir
    predictions = read_jsonl(folder / "predictions.jsonl")[:args.size]
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(folder / "retrieval.jsonl")}
    by_claim = {r["claim"]: retrieval[r["claim_id"]] for r in predictions}

    class SavedRetriever:
        fingerprint = "saved-genuine-retrieval-for-device-parity"

        def retrieve(self, claim, method="hybrid", rerank=True, keep=10):
            return RetrievalOutput(hits=[RetrievalHit.model_validate(h) for h in by_claim[claim]][:keep])

    corpus = load_corpus(cfg.path("processed_dir"))
    checker = ClaimChecker(cfg, corpus=corpus, retriever=SavedRetriever(), token_counter=lambda s: len(s.split()))
    checks = []
    for expected in predictions:
        result = checker.check(expected["claim"])
        assert result.status == "ok", result.error
        assert result.verdict == expected["verdict"]
        actual = {d.doc_id: d.judgement.verdict for d in result.documents}
        assert actual == expected["doc_verdicts"]
        assert all(q.quote == corpus[q.doc_id].sentences[q.sentence_id] for q in result.evidence)
        checks.append({"claim_id": expected["claim_id"], "verdict_matches": True,
                       "document_verdicts_match": True, "quotes_exact": True})
    write_json(args.output, {
        "device": "cpu", "torch": torch.__version__, "checks": checks,
        "config": str(args.config), "reference_run": str(folder), "model_dir": cfg.verifier.model_dir,
        "note": "Device parity check; not a new estimate of predictive accuracy.",
    })
    print(f"CPU parity and exact quotations verified for {len(checks)} fixed claims.")


if __name__ == "__main__":
    main()
