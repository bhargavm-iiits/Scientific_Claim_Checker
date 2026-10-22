import json

import pytest

from claim_checker.agents.evidence import (
    build_quotes,
    select_evidence,
    sentence_windows,
    validate_sentence_ids,
)
from claim_checker.agents.prompting import PromptLibrary
from claim_checker.config import EvidenceConfig
from claim_checker.models.ollama_client import OllamaClient
from tests.conftest import ROOT, FakeTransport


def test_quotes_are_exact_stored_sentences(corpus):
    doc = corpus["101"]
    quotes = build_quotes(doc, [2, 2, 0])
    assert [q.sentence_id for q in quotes] == [0, 2]
    assert quotes[1].quote == doc.sentences[2] and quotes[1].title == doc.title and quotes[1].doc_id == "101"
    with pytest.raises(ValueError):
        build_quotes(doc, [17])


def test_validate_sentence_ids_dedupes_and_rejects_invalid():
    valid, invalid = validate_sentence_ids([3, 1, 3, 9, -1], range(5))
    assert valid == [1, 3] and invalid == [9, -1]
    assert validate_sentence_ids([], range(3)) == ([], [])


def test_sentence_windows_are_bounded():
    sents = ["a" * 10] * 25
    wins = sentence_windows(sents, max_sentences=10, max_chars=1000)
    assert wins == [(0, 10), (10, 20), (20, 25)]
    assert sentence_windows(["x" * 50, "y" * 50], max_sentences=10, max_chars=60) == [(0, 1), (1, 2)]
    assert sentence_windows([], 10, 10) == []


def _client(cfg, responses):
    it = iter(responses)
    transport = FakeTransport(lambda p: next(it))
    return OllamaClient(cfg.model, cache_dir=None, transport=transport), transport


def test_invented_ids_are_repaired_then_dropped(cfg, corpus):
    prompts = PromptLibrary(ROOT / "prompts")
    bad = json.dumps({"sentence_ids": [2, 42], "note": "Quote: 'vitamin D cures everything'"})
    client, transport = _client(cfg, [bad, bad])
    sel, events = select_evidence(client, prompts, "Vitamin D increases bone density", corpus["101"], EvidenceConfig())
    assert sel.sentence_ids == [2] and sel.dropped_invalid_ids == [42]
    assert len(transport.calls) == 2  # one repair attempt, then bounded
    quotes = build_quotes(corpus["101"], sel.sentence_ids)
    assert quotes[0].quote == corpus["101"].sentences[2]  # model text never becomes a quote
    assert any("dropped" in e["message"] for e in events)


def test_empty_evidence_is_allowed(cfg, corpus):
    prompts = PromptLibrary(ROOT / "prompts")
    client, transport = _client(cfg, [json.dumps({"sentence_ids": [], "note": "nothing relevant"})])
    sel, _ = select_evidence(client, prompts, "Stars are made of cheese", corpus["404"], EvidenceConfig())
    assert sel.sentence_ids == [] and len(transport.calls) == 1


def test_long_abstract_is_split_into_bounded_windows(cfg, corpus):
    prompts = PromptLibrary(ROOT / "prompts")
    doc = corpus["101"].model_copy(update={"sentences": [f"Sentence {i}." for i in range(12)]})
    responses = [json.dumps({"sentence_ids": [1], "note": "a"}), json.dumps({"sentence_ids": [11], "note": "b"})]
    client, transport = _client(cfg, responses)
    sel, events = select_evidence(client, prompts, "claim", doc, EvidenceConfig(max_sentences_per_call=6))
    assert sel.windowed and sel.sentence_ids == [1, 11] and len(transport.calls) == 2
    assert "[11] Sentence 11." in transport.calls[1]["messages"][1]["content"]
