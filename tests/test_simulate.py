"""Tests for the Monte-Carlo season simulator."""
import numpy as np
import pandas as pd

from binefar_predictor.ratings import DixonColesModel
from binefar_predictor.simulate import SeasonSimulator, double_round_robin


def _model_with_teams(teams, strong=None):
    """Fit a DC model on synthetic data so all `teams` are rated."""
    rng = np.random.default_rng(1)
    strong = strong or {}
    rows = []
    ts = 1_500_000_000
    for _ in range(4):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg = rng.poisson(1.2 + strong.get(h, 0))
                ag = rng.poisson(1.0 + strong.get(a, 0))
                ts += 86400
                rows.append(dict(home=h, away=a, home_goals=int(hg),
                                 away_goals=int(ag), timestamp=ts))
    return DixonColesModel(half_life_days=0, l2=0.01).fit(pd.DataFrame(rows))


def test_double_round_robin_count():
    teams = [f"T{i}" for i in range(6)]
    fx = double_round_robin(teams)
    assert len(fx) == 6 * 5  # each hosts every other once
    assert all(h != a for h, a in fx)


def test_probabilities_are_valid():
    teams = [f"T{i}" for i in range(8)]
    model = _model_with_teams(teams)
    sim = SeasonSimulator(model, teams, target="T0", seed=3)
    res = sim.run(n_sims=2000)
    assert 0.0 <= res.p_promotion <= 1.0
    assert 0.0 <= res.p_direct <= res.p_playoff_reached + res.p_direct <= 1.0
    assert abs(res.position_dist.sum() - 1.0) < 1e-6
    assert res.position_dist.shape[0] == len(teams)


def test_stronger_target_has_higher_promotion_prob():
    teams = [f"T{i}" for i in range(8)]
    model = _model_with_teams(teams, strong={"T0": 1.5})  # T0 is dominant
    strong = SeasonSimulator(model, teams, target="T0", seed=5).run(n_sims=3000)
    weak = SeasonSimulator(model, teams, target="T7", seed=5).run(n_sims=3000)
    assert strong.p_promotion > weak.p_promotion
    assert strong.mean_position < weak.mean_position


def test_newcomer_gets_penalized_rating():
    teams = [f"T{i}" for i in range(7)] + ["Newcomer X"]
    model = _model_with_teams([f"T{i}" for i in range(7)])
    sim = SeasonSimulator(model, teams, target="T0", newcomer_penalty=0.4, seed=2)
    assert sim._attack["Newcomer X"] < 0
    assert sim._defense["Newcomer X"] > 0


def test_target_must_be_in_group():
    teams = [f"T{i}" for i in range(6)]
    model = _model_with_teams(teams)
    try:
        SeasonSimulator(model, teams, target="NotHere")
        assert False, "should have raised"
    except ValueError:
        pass
