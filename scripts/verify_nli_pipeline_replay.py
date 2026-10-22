"""Check saved development calibration against the real workflow without new inference.

Window scores are replayed exactly. Sentence-level citation scores are placeholders;
this check establishes decision parity, not citation quality or a new accuracy score.
"""
import argparse
from pathlib import Path

from analyze_verifier import decide
from transformers import AutoTokenizer

from claim_checker.config import load_config
from claim_checker.data.store import load_corpus, read_json, read_jsonl, sha256_file, write_json
from claim_checker.models.verifier import NLIVerifier
from claim_checker.retrieval.index import RetrievalOutput
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import RetrievalHit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=ROOT / "artifacts/improvement/frozen_selection_nli_title_v2.json")
    args = parser.parse_args()
    frozen_path = args.frozen
    frozen = read_json(frozen_path)
    chosen = frozen["chosen"]
    name = frozen["training_name"]
    folder = ROOT / f"artifacts/improvement/{name}_epoch{chosen['epoch']}_dev120"
    path = folder / "predictions.jsonl"
    assert read_json(folder / "status.json")["complete"] is True
    assert sha256_file(path) == chosen["source_sha256"]
    model_dir = f"artifacts/models/{name}/epoch{chosen['epoch']}"
    cfg = load_config(overrides={
        "product": {"judge_backend": "nli", "prompt_set": "v1", "judge_think": False, "evidence_mode": "full_abstract"},
        "retrieval": {"pipeline_k": chosen["k"]},
        "verifier": {"model_dir": model_dir, "include_title": frozen["include_title"],
                     "window_overlap": frozen["window_overlap"], "support_threshold": chosen["support_threshold"],
                     "contradict_threshold": chosen["contradict_threshold"], "min_retrieval_score": chosen["retrieval_gate"]},
    })
    corpus = load_corpus(cfg.path("processed_dir"))
    rows = read_jsonl(path)
    by_text = {row["claim"]: row for row in rows}
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}

    class ReplayRetriever:
        fingerprint = "saved-genuine-retrieval-decision-parity"

        def retrieve(self, claim, **kwargs):
            return RetrievalOutput(hits=[RetrievalHit.model_validate(h)
                                        for h in retrieval[by_text[claim]["claim_id"]]], timings={})

    class ReplayVerifier(NLIVerifier):
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(ROOT / model_dir, local_files_only=True)
            self.max_length, self.window_overlap, self.include_title = 512, frozen["window_overlap"], frozen["include_title"]
            self.awaiting_windows = False
            self.saved = {}
            for row in rows:
                for context in row["contexts"]:
                    if context["view"] == "abstract":
                        doc = corpus[context["doc_id"]]
                        premise = self.format_premise(doc.title, " ".join(doc.sentences[i] for i in context["sentence_ids"]))
                        self.saved[(premise, row["claim"])] = context["scores"]

        def windows(self, claim, sentences, **kwargs):
            self.awaiting_windows = True
            return super().windows(claim, sentences, **kwargs)

        def score(self, premises, hypotheses, **kwargs):
            if self.awaiting_windows:
                self.awaiting_windows = False
                # Missing keys fail: runtime must construct the exact scored contexts.
                return [self.saved[(p, h)] for p, h in zip(premises, hypotheses, strict=True)]
            # Citation ranking is outside this decision-only parity check.
            return [{"SUPPORT": 0.0, "CONTRADICT": 0.0, "NEI": 1.0} for _ in premises]

    checker = ClaimChecker(cfg, corpus=corpus, retriever=ReplayRetriever(), check_resources=False)
    checker._verifier = ReplayVerifier()
    quotations = 0
    for row in rows:
        scores = {h["doc_id"]: h["score"] for h in retrieval[row["claim_id"]]}
        contexts = [c for c in row["contexts"] if chosen["retrieval_gate"] is None
                    or scores[c["doc_id"]] >= chosen["retrieval_gate"]]
        expected = decide({"contexts": contexts}, view="abstract", k=chosen["k"],
                          threshold={"SUPPORT": chosen["support_threshold"], "CONTRADICT": chosen["contradict_threshold"]})
        result = checker.check(row["claim"], claim_id=row["claim_id"])
        assert result.status == "ok", result.error
        assert result.verdict == expected, f"Calibration/workflow mismatch for development claim {row['claim_id']}"
        assert all(q.quote == corpus[q.doc_id].sentences[q.sentence_id] for q in result.evidence)
        quotations += len(result.evidence)
    report = {"purpose": "Saved-window-score decision parity; not new inference or citation-quality evaluation",
              "claims": len(rows), "label_or_status_mismatches": 0, "source_exact_quotes": quotations,
              "predictions_sha256": sha256_file(path), "frozen_selection_sha256": sha256_file(frozen_path),
              "sentence_ranking": "Placeholders; no citation precision/recall claim is made"}
    write_json(folder / "pipeline_replay.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
