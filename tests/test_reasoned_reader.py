from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from claim_checker.agents.reasoned_reader import (
    ReaderOutput,
    comparison_problems,
    read_papers,
    reader_messages,
)
from claim_checker.errors import ModelOutputError
from claim_checker.pipeline.aggregate import aggregate
from claim_checker.schemas import CorpusDocument
from scripts.run_reasoned_reader import ReasonedChecker, configuration


def document(did="1"):
    return CorpusDocument(doc_id=did, title="Example paper", text="Stored source.",
                          sentences=["Removing A reduces B.", "This is a second sentence."],
                          source={"private_test_annotation": "must not enter prompt"})


def output(did="1", relation="supports", matched=True, ids=None):
    return {"doc_id": did, "comparison": "The intervention establishes the relation.",
            "same_relationship": matched, "relation": relation,
            "sentence_ids": [0] if ids is None else ids, "confidence": .8}


class FakeClient:
    def __init__(self, papers):
        self.parsed = ReaderOutput(papers=papers)
        self.kwargs = None
        self.stats = {"calls": 0, "cache_hits": 0}

    def structured(self, messages, schema, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(parsed=self.parsed, cached=False, thinking_chars=200,
                               eval_tokens=100, think_fallback=False, attempts=1)


def test_only_live_claim_and_source_sentences_enter_messages():
    messages = reader_messages("A promotes B.", [document()], "Only sources.")
    assert "A promotes B." in messages[1]["content"]
    assert "Removing A reduces B." in messages[1]["content"]
    assert "private_test_annotation" not in str(messages)
    assert "must not enter prompt" not in str(messages)


def test_native_thinking_and_conflict_aggregation_preserve_paper_citations():
    client = FakeClient([output("2", "opposes", ids=[1]), output("1", ids=[0, 0])])
    judgements, metadata = read_papers(client, "A promotes B.", [document("1"), document("2")], "Read sources.")
    assert [j.doc_id for j in judgements] == ["1", "2"]
    assert [j.cited_sentence_ids for j in judgements] == [[0], [1]]
    assert aggregate(judgements) == ("NEI", True)
    assert client.kwargs["think"] is True and client.kwargs["think_budget"] is None
    assert metadata["thinking_chars"] == 200


@pytest.mark.parametrize("papers", [[], [output("2")], [output(), output()],
                                   [output(ids=[9])], [output(ids=[])], [output(matched=False)]])
def test_missing_duplicate_or_ungrounded_papers_cannot_silently_become_valid(papers):
    client = FakeClient(papers)
    assert comparison_problems(client.parsed, [document()])
    with pytest.raises(ModelOutputError):
        read_papers(client, "A promotes B.", [document()], "Read sources.")


def test_insufficient_does_not_become_decisive_from_confidence():
    judgements, _ = read_papers(FakeClient([output(relation="insufficient", matched=False)]),
                                "A promotes B.", [document()], "Read sources.")
    assert judgements[0].verdict == "NEI" and judgements[0].cited_sentence_ids == []


def test_boolean_is_not_a_sentence_id():
    with pytest.raises(ValidationError):
        ReaderOutput(papers=[output(ids=[True])])


def test_empty_retrieval_needs_no_model_and_returns_no_evidence():
    assert read_papers(None, "A promotes B.", [], "Read sources.")[0] == []


def adapter(client):
    from claim_checker.schemas import RetrievalHit

    checker = ReasonedChecker(configuration(), client)
    doc = document()
    checker.resources = SimpleNamespace(
        corpus={doc.doc_id: doc}, count_claim_tokens=lambda text: 5,
        _retrieve=lambda *args: SimpleNamespace(hits=[RetrievalHit(doc_id="1", rank=1, score=3, method="hybrid")]),
    )
    return checker


def test_adapter_copies_exact_source_instead_of_generated_comparison():
    result = adapter(FakeClient([output()])).check("A promotes B.", "example")
    assert result.status == "ok" and result.verdict == "SUPPORT"
    assert result.evidence[0].quote == document().sentences[0]
    assert result.evidence[0].quote != output()["comparison"]
    assert result.metadata["retrieval_hits"][0]["doc_id"] == "1"


def test_adapter_preserves_operational_failure_as_error_not_nei():
    result = adapter(FakeClient([])).check("A promotes B.", "example")
    assert result.status == "error" and result.verdict is None
    assert result.error.error_type == "ModelOutputError"
