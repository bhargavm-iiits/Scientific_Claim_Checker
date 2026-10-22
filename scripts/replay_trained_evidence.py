"""Check full-workflow parity on saved development outputs, with no model calls."""
from pathlib import Path

from transformers import AutoTokenizer

from claim_checker.agents.judge_nli import choose_label
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
from claim_checker.models.evidence_classifier import select_sentence_ids
from claim_checker.models.ollama_client import OllamaClient
from claim_checker.models.verifier import NLIVerifier
from claim_checker.pipeline.aggregate import aggregate_verdicts
from claim_checker.retrieval.index import RetrievalOutput
from claim_checker.runtime import ClaimChecker
from claim_checker.schemas import RetrievalHit

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "artifacts/improvement/trained_evidence_deberta_dev120_v2"
    chosen = read_json(folder / "selector_calibration_refinement.json")["chosen"]
    rows = read_jsonl(folder / "predictions.jsonl")
    by_text = {}
    cfg = load_config(ROOT / "configs/trained_evidence_candidate.yaml")
    corpus, gold = load_corpus(cfg.path("processed_dir")), load_gold(cfg.path("processed_dir"), "train")
    family = read_json(ROOT / "artifacts/improvement/frozen_selection_nli_title_family_v1.json")
    model = ROOT / f"artifacts/models/{family['training_name']}/epoch{family['chosen']['epoch']}"
    retrieval = {r["claim_id"]: r["hits"] for r in read_jsonl(ROOT / "artifacts/runs/dev_v1/retrieval.jsonl")}
    for row in rows:
        by_text[gold[row["claim_id"]].claim] = row

    class ReplayRetriever:
        fingerprint = "saved-genuine-retrieval-integration-parity"

        def retrieve(self, claim, **kwargs):
            return RetrievalOutput(hits=[RetrievalHit.model_validate(h) for h in retrieval[by_text[claim]["claim_id"]]], timings={})

    class ReplaySelector:
        def score(self, claim, doc):
            return next(d["selector_probabilities"] for d in by_text[claim]["documents"] if d["doc_id"] == doc.doc_id)

    class ReplayReader(NLIVerifier):
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            self.include_title, self.window_overlap, self.max_length = True, 2, 512
            self.saved = {}
            for row in rows:
                claim = gold[row["claim_id"]].claim
                for document in row["documents"]:
                    doc = corpus[document["doc_id"]]
                    for context in document["contexts"]:
                        sentences = [doc.sentences[i] for i in context["ids"]]
                        windows = self.windows(claim, sentences, title=doc.title)
                        for window, score in zip(windows, context["window_scores"], strict=True):
                            premise = self.format_premise(doc.title, " ".join(sentences[i] for i in window))
                            self.saved[(claim, premise)] = score

        def score(self, premises, hypotheses, **kwargs):
            return [self.saved[(h, p)] for p, h in zip(premises, hypotheses, strict=True)]

    class ForbiddenTransport:
        def chat(self, *args, **kwargs):
            raise AssertionError("A local replay must never call an Ollama model")

        def list_models(self):
            raise AssertionError("A local replay must never query Ollama models")

        def version(self):
            raise AssertionError("A local replay must never query Ollama")

    client = OllamaClient(cfg.model, cache_dir=None, transport=ForbiddenTransport())
    checker = ClaimChecker(cfg, corpus=corpus, retriever=ReplayRetriever(), llm=client, check_resources=False)
    checker._evidence_selector, checker._verifier = ReplaySelector(), ReplayReader()
    records = []
    for row in rows:
        labels = []
        for document in row["documents"]:
            if document["rank"] > chosen["k"] or document["retrieval_score"] < chosen["retrieval_gate"]:
                continue
            ids = select_sentence_ids(document["selector_probabilities"], threshold=chosen["selector_threshold"], maximum=chosen["max_sentences"])
            if ids:
                context = next(c for c in document["contexts"] if c["ids"] == ids)
                labels.extend(choose_label(s, {"SUPPORT": chosen["support_threshold"], "CONTRADICT": chosen["contradict_threshold"]})
                              for s in context["window_scores"])
        expected = aggregate_verdicts(labels)[0]
        result = checker.check(gold[row["claim_id"]].claim, claim_id=row["claim_id"])
        assert result.status == "ok", result.error
        assert result.verdict == expected, f"Integration/calibration decision mismatch: {row['claim_id']}"
        assert all(q.quote == corpus[q.doc_id].sentences[q.sentence_id] for q in result.evidence)
        records.append(prediction_record("replay_only", gold[row["claim_id"]], result))
    report = {"purpose": "Saved-output integration parity; not fresh predictive accuracy", "claims": len(rows),
              "decision_or_status_mismatches": 0, "model_calls": client.stats["calls"],
              "source_predictions_sha256": sha256_file(folder / "predictions.jsonl"),
              "selected_evidence_metrics_on_saved_outputs": evidence_metrics(records, "selected"),
              "cited_evidence_metrics_on_saved_outputs": evidence_metrics(records, "cited"),
              "source_quote_validity": quote_validity(records, corpus)}
    write_json(folder / "pipeline_replay.json", report)
    print({key: report[key] for key in ("claims", "decision_or_status_mismatches", "model_calls", "source_quote_validity")}, flush=True)


if __name__ == "__main__":
    main()
