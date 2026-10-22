import pytest

from claim_checker.models.passage_windows import best_passage, passage_windows


class WordTokenizer:
    def __call__(self, premise, claim, **kwargs):
        return {"input_ids": list(range(len(premise.split()) + len(claim.split()) + 4))}


def test_oversize_pairs_split_without_losing_the_final_sentence():
    sentences = ["first " * 6, "last " * 6]
    windows = passage_windows(WordTokenizer(), "claim", "title", sentences, width=2, max_length=15)
    assert windows == [[0], [1]]
    assert {i for group in windows for i in group} == {0, 1}


def test_oversize_single_sentence_is_rejected_instead_of_silently_truncated():
    with pytest.raises(ValueError, match="token budget"):
        passage_windows(WordTokenizer(), "claim", "title", ["word " * 30], width=1, max_length=15)


def test_close_opposing_passages_abstain_and_a_clear_winner_keeps_its_index():
    scores = [{"SUPPORT": 0.94, "CONTRADICT": 0.01, "NEI": 0.05},
              {"SUPPORT": 0.02, "CONTRADICT": 0.92, "NEI": 0.06}]
    assert best_passage(scores, support_threshold=0.9, contradict_threshold=0.9, margin=0.1) == ("NEI", None)
    assert best_passage(scores, support_threshold=0.9, contradict_threshold=0.9, margin=0) == ("SUPPORT", 0)
    assert best_passage(scores, support_threshold=0.98, contradict_threshold=0.9, margin=0) == ("NEI", None)
