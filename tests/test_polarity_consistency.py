import math

import pytest

from scripts.run_polarity_consistency import SWAP, ConsistentVerifier, consistent_scores, negate


def test_equivariant_inputs_are_preserved():
    original = {"SUPPORT": .8, "CONTRADICT": .05, "NEI": .15}
    negated = {label: original[SWAP[label]] for label in SWAP}
    assert consistent_scores(original, negated) == pytest.approx(original)


def test_swapping_hypotheses_swaps_support_and_contradiction():
    first = {"SUPPORT": .7, "CONTRADICT": .2, "NEI": .1}
    second = {"SUPPORT": .3, "CONTRADICT": .4, "NEI": .3}
    forward = consistent_scores(first, second)
    reverse = consistent_scores(second, first)
    assert forward == pytest.approx({label: reverse[SWAP[label]] for label in SWAP})
    assert sum(forward.values()) == pytest.approx(1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_model_probability_cannot_enter_pooling(bad):
    probabilities = {"SUPPORT": .4, "CONTRADICT": .3, "NEI": .3}
    with pytest.raises(ValueError):
        consistent_scores({**probabilities, "SUPPORT": bad}, probabilities)


def test_zero_softmax_underflow_has_finite_normalized_result():
    scores = consistent_scores({"SUPPORT": 1., "CONTRADICT": 0., "NEI": 0.},
                               {"SUPPORT": 1., "CONTRADICT": 0., "NEI": 0.})
    assert all(math.isfinite(value) for value in scores.values())
    assert sum(scores.values()) == pytest.approx(1)


def test_negation_preserves_gene_case_and_original_quantifiers():
    assert negate("RAD52 affects all B cells.") == "It is not the case that RAD52 affects all B cells."
    with pytest.raises(ValueError):
        negate(" ")


def test_cached_ordinary_scores_do_not_cache_the_enabled_mode(monkeypatch):
    from claim_checker.models.verifier import NLIVerifier

    calls = []

    def fake_score(self, premises, hypotheses, batch_size):
        calls.extend(zip(premises, hypotheses, strict=True))
        return [{"SUPPORT": .1, "CONTRADICT": .8, "NEI": .1} if h.startswith("It is not")
                else {"SUPPORT": .8, "CONTRADICT": .1, "NEI": .1} for h in hypotheses]

    monkeypatch.setattr(NLIVerifier, "score", fake_score)
    scorer = ConsistentVerifier.__new__(ConsistentVerifier)
    scorer.memo, scorer.enabled = {}, False
    ordinary = scorer.score(["source"], ["claim"])
    scorer.enabled = True
    pooled = scorer.score(["source"], ["claim"])
    assert ordinary[0] == pytest.approx(pooled[0])
    scorer.enabled = False
    assert scorer.score(["source"], ["claim"]) == ordinary
    assert calls == [("source", "claim"), ("source", negate("claim"))]


def test_control_and_candidate_share_the_longer_hypothesis_window_budget(monkeypatch):
    from claim_checker.models.verifier import NLIVerifier

    claims = []
    monkeypatch.setattr(NLIVerifier, "windows", lambda self, claim, *args, **kwargs: claims.append(claim) or [[0]])
    scorer = ConsistentVerifier.__new__(ConsistentVerifier)
    scorer.tokenizer = type("Tokenizer", (), {"encode": lambda self, text, **kwargs: text.split()})()
    scorer.enabled = False
    scorer.windows("A promotes B.", ["source"])
    scorer.enabled = True
    scorer.windows("A promotes B.", ["source"])
    assert claims == [negate("A promotes B.")] * 2
