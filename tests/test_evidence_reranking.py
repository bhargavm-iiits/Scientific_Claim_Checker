import pytest

from claim_checker.data.evidence_reranking import labeled_windows
from tests.conftest import make_doc


class WordTokenizer:
    def __call__(self, first, second=None, add_special_tokens=True):
        count = len(first.split()) + (len(second.split()) if second is not None else 0)
        count += (3 if second is not None else 2) if add_special_tokens else 0
        return {"input_ids": [0] * count}


def test_ranking_supervision_keeps_complete_evidence_and_omits_partial_rationales():
    sentences = [" ".join([f"word{i}"] * 9) + "." for i in range(3)]
    doc = make_doc("1", "Findings", sentences)
    rows, omissions = labeled_windows(WordTokenizer(), "test claim", doc, reference=True,
                                      alternatives=[[0, 1]], max_length=28)
    assert len(rows) == 1 and rows[0]["target"] == 1
    assert rows[0]["sentence_ids"] == [0, 1]
    assert rows[0]["premise"] == f"Findings\n{sentences[0]} {sentences[1]}"
    assert omissions == [{"reason": "partial_rationale", "sentence_ids": [1, 2]}]
    assert len(WordTokenizer()("test claim", rows[0]["premise"])["input_ids"]) <= 28


@pytest.mark.parametrize("reference, expected", [(False, "weak_nonreference"), (True, "annotated_NEI")])
def test_negative_ranking_targets_keep_weak_and_annotated_sources_distinct(reference, expected):
    doc = make_doc("2", "A source title", ["The original source sentence remains unchanged."])
    rows, omissions = labeled_windows(WordTokenizer(), "test claim", doc, reference=reference, alternatives=[])
    assert not omissions and rows[0]["target"] == 0 and rows[0]["supervision"] == expected
    assert doc.sentences[0] in rows[0]["premise"]


def test_oversized_positive_rationale_is_recorded_instead_of_trained_with_truncated_input():
    doc = make_doc("3", "Title", [" ".join(["original"] * 50)])
    rows, omissions = labeled_windows(WordTokenizer(), "test claim", doc, reference=True, alternatives=[[0]], max_length=28)
    assert not rows and omissions[0]["reason"] == "oversized_sentence"


def test_ranking_builder_rejects_invalid_annotations():
    doc = make_doc("3", "Title", ["An original source sentence."])
    with pytest.raises(ValueError, match="outside the source"):
        labeled_windows(WordTokenizer(), "test claim", doc, reference=True, alternatives=[[7]])


def test_ranking_builder_rejects_claim_without_sufficient_context_budget():
    doc = make_doc("3", "Title", ["An original source sentence."])
    with pytest.raises(ValueError, match="budget"):
        labeled_windows(WordTokenizer(), " ".join(["claim"] * 20), doc, reference=True, alternatives=[], max_length=28)
