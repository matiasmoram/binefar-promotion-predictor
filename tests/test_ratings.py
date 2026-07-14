"""Tests for the rating models."""
import numpy as np
import pandas as pd
import pytest

from binefar_predictor.ratings import DixonColesModel, EloModel


def _synthetic_matches(n_seasons=3, seed=0):
    """A synthetic league where team A is strong and team D is weak.

    Goals are drawn from Poisson with team-specific rates so the fitted model
    should recover the correct strength ordering.
    """
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    strength = {"A": 1.9, "B": 1.3, "C": 1.0, "D": 0.6}  # expected goals scale
    rows = []
    ts = 1_500_000_000
    for _ in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg = rng.poisson(strength[h] * 1.15)  # home boost
                ag = rng.poisson(strength[a] * 0.9)
                ts += 86400 * 3
                rows.append(
                    dict(home=h, away=a, home_goals=int(hg), away_goals=int(ag),
                         timestamp=ts)
                )
    return pd.DataFrame(rows)


def test_dixoncoles_recovers_ordering():
    m = _synthetic_matches()
    model = DixonColesModel(half_life_days=0, l2=0.01).fit(m)  # no time decay
    tbl = model.strength_table()
    order = list(tbl["team"])
    assert order[0] == "A", f"expected A strongest, got {order}"
    assert order[-1] == "D", f"expected D weakest, got {order}"


def test_dixoncoles_probabilities_sum_to_one():
    m = _synthetic_matches()
    model = DixonColesModel(l2=0.01).fit(m)
    for h, a in [("A", "D"), ("D", "A"), ("B", "C")]:
        ph, pd_, pa = model.win_probabilities(h, a)
        assert abs(ph + pd_ + pa - 1.0) < 1e-9
        assert 0 <= ph <= 1 and 0 <= pd_ <= 1 and 0 <= pa <= 1


def test_dixoncoles_home_advantage_positive():
    m = _synthetic_matches()
    model = DixonColesModel(l2=0.01).fit(m)
    assert model.home > 0  # home teams score more on average


def test_stronger_team_more_likely_to_win():
    m = _synthetic_matches()
    model = DixonColesModel(l2=0.01).fit(m)
    ph_strong, _, _ = model.win_probabilities("A", "D")
    ph_weak, _, _ = model.win_probabilities("D", "A")
    assert ph_strong > ph_weak


def test_scoreline_matrix_normalized():
    m = _synthetic_matches()
    model = DixonColesModel(l2=0.01).fit(m)
    mat = model.scoreline_matrix("A", "B")
    assert abs(mat.sum() - 1.0) < 1e-9
    assert (mat >= 0).all()


def test_elo_orders_teams():
    m = _synthetic_matches()
    elo = EloModel().fit(m)
    assert elo.ratings["A"] > elo.ratings["D"]


def test_unknown_team_defaults_to_average():
    m = _synthetic_matches()
    model = DixonColesModel(l2=0.01).fit(m)
    a, d = model._params("NEVER_SEEN")
    assert a == 0.0 and d == 0.0
