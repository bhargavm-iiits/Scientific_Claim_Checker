import pytest

from scripts.train_calibrated_evidence_ranker import lower_tail_gate


def test_evidence_gate_uses_fixed_order_statistic_independently_of_score_order():
    scores = list(range(39))
    assert lower_tail_gate(scores) == 1
    assert lower_tail_gate(list(reversed(scores))) == 1


def test_small_calibration_set_retains_all_instead_of_claiming_precise_95percent_coverage():
    assert lower_tail_gate([0.1, 0.3, 0.5]) is None


@pytest.mark.parametrize("scores, alpha", [([], 0.05), ([float("nan")], 0.05), ([float("inf")], 0.05), ([1], 0), ([1], 1)])
def test_calibration_rejects_invalid_scores_or_error_rates(scores, alpha):
    with pytest.raises(ValueError):
        lower_tail_gate(scores, alpha)
