from claim_checker.data.verifier_context import positive_window


class WordTokenizer:
    def __call__(self, premise, claim, **kwargs):
        return {"input_ids": [0] * (len(premise.split()) + len(claim.split()) + 4)}


def test_positive_window_keeps_late_rationale_instead_of_truncating_it():
    sentences = ["word " * 8 for _ in range(12)]
    sentences[-1] = "The measured treatment reduced the outcome."
    result = positive_window(WordTokenizer(), "Treatment changes outcome", "Title", sentences, [[11]], max_length=40)
    assert 11 in result["context_sentence_ids"]
    assert result["rationale_sentence_ids"] == [11]
    assert sentences[-1] in result["premise"]
    assert len(WordTokenizer()(result["premise"], "Treatment changes outcome")["input_ids"]) <= 40


def test_positive_window_does_not_renumber_original_rationale():
    sentences = ["long " * 30, "first finding", "long " * 30, "second finding"]
    result = positive_window(WordTokenizer(), "A claim", "A title", sentences, [[1, 3]], max_length=20, include_title=True)
    assert result["context_sentence_ids"] == [1, 3]
    assert result["rationale_sentence_ids"] == [1, 3]
    assert "Title: A title" in result["premise"]


def test_unfit_rationale_is_explicitly_omitted():
    result = positive_window(WordTokenizer(), "claim", "title", ["word " * 100], [[0]], max_length=20)
    assert result is None
