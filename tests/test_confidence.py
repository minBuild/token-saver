import pytest

from token_saver.confidence import choice_confidence, noul_confidence, score_confidence


def test_choice_confidence_matches_documented_examples():
    # Only the top probability counts, so these two distributions agree.
    assert choice_confidence([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert choice_confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)


def test_choice_confidence_bounds():
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert choice_confidence([1.0]) == 1.0


def test_score_confidence_penalises_distance_from_the_top_level():
    # Mass on a neighbouring level is a mild disagreement...
    assert score_confidence([0.0, 0.5, 0.5]) == pytest.approx(0.25)
    # ...mass on the far end is a real one, floored at zero.
    assert score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0)
    assert score_confidence([0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_noul_confidence_is_distance_from_a_coin_flip():
    assert noul_confidence(0.5) == pytest.approx(0.0)
    assert noul_confidence(0.9) == pytest.approx(0.8)
    assert noul_confidence(0.1) == pytest.approx(0.8)
