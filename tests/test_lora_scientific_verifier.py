"""Adapter gradients/merging and source-grounded fallback; no remote models."""
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from claim_checker.schemas import CorpusDocument, DocumentJudgement, RetrievalHit
from scripts.lora_scientific_verifier import (
    FeedbackChecker,
    adapt_encoder,
    pick_rescues,
    promote_hits,
    row_weight,
    weak_sentence_negatives,
)
from scripts.run_lora_scientific_verifier import experiment_config, quality_target


@pytest.fixture(autouse=True, params=("v1", "v2"))
def implementation(request, monkeypatch):
    suffix = "_v2" if request.param == "v2" else ""
    module = importlib.import_module("scripts.lora_scientific_verifier" + suffix)
    for name in ("FeedbackChecker", "adapt_encoder", "pick_rescues", "promote_hits", "row_weight", "weak_sentence_negatives"):
        monkeypatch.setattr(sys.modules[__name__], name, getattr(module, name))
    runner = importlib.import_module("scripts.run_lora_scientific_verifier" + suffix)
    for name in ("experiment_config", "quality_target"):
        monkeypatch.setattr(sys.modules[__name__], name, getattr(runner, name))


def tiny_base():
    return DebertaV2ForSequenceClassification(DebertaV2Config(
        vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=24, max_position_embeddings=32, hidden_dropout_prob=0.,
        attention_probs_dropout_prob=0., pooler_dropout=0., cls_dropout=0., num_labels=3,
    ))


def test_all_layers_adapt_and_frozen_embeddings_stay_frozen():
    base = tiny_base().eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    original = base(**inputs).logits.detach().clone()
    model = adapt_encoder(base, rank=2, alpha=4, dropout=0.).eval()
    torch.testing.assert_close(model(**inputs).logits, original)
    torch.nn.functional.cross_entropy(model(**inputs).logits, torch.tensor([2])).backward()
    for layer in range(2):
        for projection in ("query_proj", "value_proj"):
            parameters = [(name, p) for name, p in model.named_parameters()
                          if f"encoder.layer.{layer}." in name and projection in name and ".lora_B." in name]
            assert len(parameters) == 1 and parameters[0][1].grad.abs().sum() > 0
    assert all(p.grad is None for name, p in model.named_parameters() if "word_embeddings" in name)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B." in name:
                p.add_(.1)
    before_merge = model(**inputs).logits.detach()
    merged = model.merge_and_unload(safe_merge=True).eval()
    torch.testing.assert_close(merged(**inputs).logits, before_merge, atol=1e-6, rtol=1e-5)


def hit(did, rank):
    return RetrievalHit(doc_id=did, rank=rank, score=1., method="hybrid")


def judgement(did, verdict="NEI", confidence=None, flags=None):
    return DocumentJudgement(doc_id=did, verdict=verdict, confidence=confidence,
                             cited_sentence_ids=[] if verdict == "NEI" else [0], consistency_flags=flags or [])


def test_opposing_rescue_evidence_is_retained():
    candidates = [(hit("4", 4), judgement("4", "SUPPORT", .96)),
                  (hit("5", 5), judgement("5", "CONTRADICT", .98)),
                  (hit("6", 6), judgement("6", "SUPPORT", .91))]
    selected = pick_rescues([judgement("1")], candidates)
    assert [h.doc_id for h, _ in selected] == ["4", "5"]
    assert pick_rescues([judgement("1", "SUPPORT", .6)], candidates) == []
    assert pick_rescues([judgement("1", flags=["window_conflict"])], candidates) == []


def test_low_confidence_cannot_become_a_rescue_and_window_conflict_is_reported():
    assert pick_rescues([], [(hit("4", 4), judgement("4", "SUPPORT", .89))]) == []
    candidates = [(hit("4", 4), judgement("4", "SUPPORT", .99)),
                  (hit("5", 5), judgement("5", flags=["window_conflict"]))]
    assert pick_rescues([], candidates)[0][0].doc_id == "5"


def test_promotion_keeps_only_genuinely_retrieved_sources():
    hits = [hit(str(i), i) for i in range(1, 7)]
    result = promote_hits(hits, ["5", "6"])
    assert [h.doc_id for h in result[:3]] == ["1", "5", "6"]
    assert {h.doc_id for h in result} == {h.doc_id for h in hits}
    assert [h.rank for h in result] == list(range(1, 7))
    assert [h.rank for h in hits] == list(range(1, 7))
    with pytest.raises(ValueError, match="genuinely retrieved"):
        promote_hits(hits, ["99"])
    with pytest.raises(ValueError, match="distinct"):
        promote_hits(hits, ["5", "5"])


def test_weak_sentence_training_never_relabels_an_annotated_sentence():
    doc = CorpusDocument(doc_id="7", title="Trial", text="", sentences=["Drug reduces risk.", "Drug dosage was measured.", "Another positive result."])
    gold = {"1": SimpleNamespace(evidence_by_doc={"7": [0, 2]})}
    row = {"claim_id": "1", "doc_id": "7", "claim": "Drug reduces risk", "label": "SUPPORT"}
    negatives = weak_sentence_negatives([row, row], {"7": doc}, gold)
    assert len(negatives) == 1
    assert negatives[0]["context_sentence_ids"] == [1]
    assert negatives[0]["premise"] == "Title: Trial\nAbstract: Drug dosage was measured."
    assert row_weight(negatives[0]) == .1
    assert row_weight({"view": "hard_negative"}) == .25
    assert row_weight({"view": "rationale"}) == 1.


def fake_checker(monkeypatch, cfg, labels):
    module = importlib.import_module(FeedbackChecker.__module__)

    checker = FeedbackChecker.__new__(FeedbackChecker)
    checker.cfg, checker.feedback, checker.identity = cfg, True, {}
    docs = {str(i): CorpusDocument(doc_id=str(i), title=f"Paper{i}", text="Source.", sentences=["Source."])
            for i in range(1, 6)}
    checker.resources = SimpleNamespace(corpus=docs, count_claim_tokens=lambda text: 4,
                                        _retrieve=lambda *args: SimpleNamespace(hits=[hit(str(i), i) for i in range(1, 6)]))
    checker.verifier = None
    calls = []

    def reader(verifier, claim, doc, context, threshold):
        calls.append((claim, doc.doc_id, context))
        return labels.get(doc.doc_id, judgement(doc.doc_id))

    monkeypatch.setattr(module, "judge_document" if hasattr(module, "judge_document") else "judge_nli", reader)
    return checker, calls


def test_pipeline_fallback_preserves_conflicting_sources(monkeypatch, cfg):
    checker, calls = fake_checker(monkeypatch, cfg, {"4": judgement("4", "SUPPORT", .95), "5": judgement("5", "CONTRADICT", .96)})
    result = checker.check("A source-based claim", "opaque-runtime-id")
    assert result.status == "ok" and result.verdict == "NEI" and result.conflict
    assert result.metadata["rescued_doc_ids"] == ["4", "5"]
    assert len(result.documents) == 3 and len(result.evidence) == 2
    assert {q.doc_id for q in result.evidence} == {"4", "5"}
    assert all(q.quote == "Source." for q in result.evidence)
    assert len(calls) == 5
    assert all(c[0] == "A source-based claim" for c in calls)


def test_pipeline_does_not_rescue_after_a_decisive_initial_paper(monkeypatch, cfg):
    checker, calls = fake_checker(monkeypatch, cfg, {"1": judgement("1", "SUPPORT", .75)})
    result = checker.check("A source-based claim")
    assert result.status == "ok" and result.verdict == "SUPPORT"
    assert not result.metadata["rescue_attempted"] and len(calls) == 3


def test_failure_is_an_operational_error(monkeypatch, cfg):
    checker, _ = fake_checker(monkeypatch, cfg, {})
    checker.resources._retrieve = lambda *args: (_ for _ in ()).throw(RuntimeError("retrieval failed"))
    result = checker.check("A claim")
    assert result.status == "error" and result.verdict is None


def test_all_scores_target_does_not_hide_a_bad_class_or_evidence_score():
    values = {"precision": .99, "recall": .99, "f1": .99}
    metrics = {"verdict": {"accuracy": .99, "macro_f1": .99, "per_class": {"SUPPORT": values, "CONTRADICT": values, "NEI": values}},
               "evidence_cited": {"precision": .99, "recall": .94}, "retrieval": {"recall@3": .99, "recall@10": .99},
               "quotes": {"validity": 1.}}
    assert quality_target(metrics)["primary_accuracy_and_macro_above_0_95"]
    assert not quality_target(metrics)["all_above_0_95"]
    metrics["evidence_cited"]["recall"] = .99
    assert quality_target(metrics)["all_above_0_95"]


def test_wider_retrieval_is_local_to_experiment():
    from claim_checker.config import load_config

    cfg = experiment_config()
    assert (cfg.retrieval.candidates_per_method, cfg.retrieval.rerank_top_n, cfg.retrieval.keep_top) == (100, 50, 20)
    original = load_config()
    assert (original.retrieval.candidates_per_method, original.retrieval.rerank_top_n, original.retrieval.keep_top) == (30, 20, 10)
