import pytest

from claim_checker.models.evidence_classifier import candidate_context, select_sentence_ids


class WordTokenizer:
    def __init__(self):
        self.pairs = []

    def __call__(self, first, second, **kwargs):
        self.pairs.append((first, second))
        return {"input_ids": [0] * (len(first.split()) + len(second.split()) + 4)}


def test_candidate_only_context_preserves_source_and_premise_first_order():
    tokenizer = WordTokenizer()
    sentences = ["Background.", "The measured outcome decreased.", "Other context."]
    premise, ids = candidate_context(tokenizer, "A claim", "Paper title", sentences, 1,
                                     use_neighbors=False, input_order="premise_first")
    assert ids == [1] and sentences[1] in premise
    assert sentences[0] not in premise and sentences[2] not in premise
    assert tokenizer.pairs[-1] == (premise, "A claim")


def test_candidate_does_not_silently_truncate_an_oversized_source_sentence():
    with pytest.raises(ValueError, match="token budget"):
        candidate_context(WordTokenizer(), "claim", "title", ["word " * 100], 0, max_length=20)


def test_selected_indices_return_to_source_order_and_low_scores_abstain():
    assert select_sentence_ids([0.8, 0.1, 0.9], threshold=0.5, maximum=2) == [0, 2]
    assert select_sentence_ids([0.8, 0.1, 0.9], threshold=0.95, maximum=2) == []
    assert select_sentence_ids([0.8, 0.8, 0.8], threshold=0.5, maximum=2) == [0, 1]
