"""Tests for pi-ratings, independent-Poisson toggle, and the ensemble."""
import numpy as np
import pandas as pd

from binefar_predictor.ratings import DixonColesModel, PiRatingModel


def _synthetic(seed=0, n=4):
    rng = np.random.default_rng(seed)
    teams = [chr(ord("A") + i) for i in range(n)]
    strength = {t: 1.8 - 0.3 * i for i, t in enumerate(teams)}
    rows, ts = [], 1_500_000_000
    for _ in range(4):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                ts += 86400
                rows.append(dict(home=h, away=a,
                                 home_goals=int(rng.poisson(strength[h] * 1.1)),
                                 away_goals=int(rng.poisson(strength[a] * 0.9)),
                                 timestamp=ts))
    return pd.DataFrame(rows)


def test_pi_ratings_order_teams():
    pi = PiRatingModel().fit(_synthetic())
    tbl = pi.strength_table()
    assert tbl.iloc[0]["team"] == "A"
    assert tbl.iloc[-1]["team"] == "D"


def test_independent_poisson_has_zero_rho():
    m = _synthetic()
    dc0 = DixonColesModel(l2=0.01, fix_rho=0.0).fit(m)
    assert abs(dc0.rho) < 1e-9
    dc = DixonColesModel(l2=0.01).fit(m)
    # a normal fit is free to move rho away from exactly 0
    assert dc.rho != 0.0


def test_independent_poisson_still_orders_teams():
    dc0 = DixonColesModel(l2=0.01, fix_rho=0.0).fit(_synthetic())
    tbl = dc0.strength_table()
    assert tbl.iloc[0]["team"] == "A"
