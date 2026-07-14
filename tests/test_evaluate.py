"""Tests for the scoring metrics."""
import numpy as np

from binefar_predictor.evaluate import brier, log_loss, rps


def test_perfect_prediction_scores_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outc = np.array([0, 1, 2])
    assert brier(probs, outc) < 1e-12
    assert rps(probs, outc) < 1e-12
    assert log_loss(probs, outc) < 1e-6  # clipped, so ~0


def test_uniform_prediction_brier():
    probs = np.full((100, 3), 1 / 3)
    outc = np.zeros(100, dtype=int)
    # Brier for uniform vs one-hot = (2/3)^2 + (1/3)^2 + (1/3)^2 = 0.6667
    assert abs(brier(probs, outc) - (4 / 9 + 1 / 9 + 1 / 9)) < 1e-9


def test_confident_wrong_worse_than_uniform_logloss():
    conf_wrong = np.array([[0.001, 0.001, 0.998]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    outc = np.array([0])
    assert log_loss(conf_wrong, outc) > log_loss(uniform, outc)


def test_rps_rewards_closeness_for_ordinal():
    # true outcome = away (2). A "draw" guess should score better than a
    # "home" guess because draw is ordinally closer to away.
    close = np.array([[0.0, 1.0, 0.0]])   # predicts draw
    far = np.array([[1.0, 0.0, 0.0]])     # predicts home
    outc = np.array([2])
    assert rps(close, outc) < rps(far, outc)
