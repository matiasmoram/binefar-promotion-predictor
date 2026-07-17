"""Descriptive analytics and sensitivity analysis.

Two things:

* :func:`form_tables` — descriptive splits for the target club from the match
  data (home/away, rolling form, per-season points), useful context that does
  not depend on any model.
* :func:`sensitivity` — how the headline promotion probability moves as the
  model's judgement-call parameters vary (rating half-life, L2 shrinkage,
  newcomer penalty, national-phase conversion). This makes the model's
  assumptions transparent instead of hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .ratings import DixonColesModel
from .simulate import SeasonSimulator


def form_tables(matches: pd.DataFrame, team: str) -> dict:
    """Home/away and per-season summaries for one club."""
    tm = matches[(matches["home"] == team) | (matches["away"] == team)].copy()
    if tm.empty:
        return {}

    def pts(row):
        gd = row["home_goals"] - row["away_goals"]
        if row["home"] == team:
            return 3 if gd > 0 else (1 if gd == 0 else 0)
        return 3 if gd < 0 else (1 if gd == 0 else 0)

    def gf_ga(row):
        if row["home"] == team:
            return row["home_goals"], row["away_goals"]
        return row["away_goals"], row["home_goals"]

    tm["points"] = tm.apply(pts, axis=1)
    tm[["gf", "ga"]] = tm.apply(lambda r: pd.Series(gf_ga(r)), axis=1)
    tm["venue"] = np.where(tm["home"] == team, "home", "away")

    by_venue = (
        tm.groupby("venue")
        .agg(matches=("points", "size"), ppg=("points", "mean"),
             gf=("gf", "mean"), ga=("ga", "mean"))
        .round(2)
        .to_dict("index")
    )
    by_season = (
        tm.groupby("season")
        .agg(matches=("points", "size"), points=("points", "sum"),
             gf=("gf", "sum"), ga=("ga", "sum"))
        .to_dict("index")
    )
    last10 = tm.sort_values("timestamp").tail(10)
    return {
        "by_venue": by_venue,
        "by_season": by_season,
        "last10_ppg": round(float(last10["points"].mean()), 2),
        "last10_record": {
            "W": int((last10["points"] == 3).sum()),
            "D": int((last10["points"] == 1).sum()),
            "L": int((last10["points"] == 0).sum()),
        },
    }


def whatif_curve(
    matches: pd.DataFrame,
    latest_standings: pd.DataFrame,
    target: str = config.CLUB_NAME,
    shifts: list[float] | None = None,
    n_sims: int = 12_000,
) -> list[dict]:
    """Promotion probability vs a squad-strength adjustment of the target club.

    Drives the dashboard "what-if" slider: how would a stronger/weaker Binéfar
    squad than last season's form implies change the promotion odds? Each point
    shifts the club's net Dixon-Coles rating by ``delta`` and re-simulates.
    """
    from . import data as _data

    shifts = shifts if shifts is not None else [round(x / 10, 2) for x in range(-4, 7)]
    group = _data.project_target_group(latest_standings)
    model = DixonColesModel(half_life_days=365, l2=0.05).fit(matches)
    curve = []
    for d in shifts:
        res = SeasonSimulator(
            model, group, target=target, target_strength_shift=d
        ).run(n_sims=n_sims)
        curve.append({
            "shift": d,
            "p_promotion": round(res.p_promotion, 4),
            "mean_position": round(res.mean_position, 2),
        })
    return curve


def sensitivity(
    matches: pd.DataFrame,
    latest_standings: pd.DataFrame,
    target: str = config.CLUB_NAME,
    n_sims: int = 15_000,
) -> pd.DataFrame:
    """Vary one assumption at a time; report the promotion probability.

    Returns a tidy DataFrame (parameter, value, p_promotion).
    """
    from . import data as _data

    group = _data.project_target_group(latest_standings)
    rows = []

    # 1) rating half-life
    for hl in [180, 365, 540, 730]:
        model = DixonColesModel(half_life_days=hl, l2=0.05).fit(matches)
        res = SeasonSimulator(model, group, target=target).run(n_sims=n_sims)
        rows.append({"parameter": "half_life_days", "value": hl,
                     "p_promotion": round(res.p_promotion, 4)})

    # baseline model reused for the cheaper knobs
    base = DixonColesModel(half_life_days=365, l2=0.05).fit(matches)

    # 2) L2 shrinkage
    for l2 in [0.0, 0.05, 0.15, 0.3]:
        model = DixonColesModel(half_life_days=365, l2=l2).fit(matches)
        res = SeasonSimulator(model, group, target=target).run(n_sims=n_sims)
        rows.append({"parameter": "l2", "value": l2,
                     "p_promotion": round(res.p_promotion, 4)})

    # 3) newcomer penalty
    for pen in [0.0, 0.25, 0.5, 0.75]:
        res = SeasonSimulator(base, group, target=target,
                              newcomer_penalty=pen).run(n_sims=n_sims)
        rows.append({"parameter": "newcomer_penalty", "value": pen,
                     "p_promotion": round(res.p_promotion, 4)})

    # 4) national-phase conversion
    for conv in [0.2, 0.3, 0.4, 0.55, 0.7]:
        res = SeasonSimulator(base, group, target=target,
                              national_conversion=conv).run(n_sims=n_sims)
        rows.append({"parameter": "national_conversion", "value": conv,
                     "p_promotion": round(res.p_promotion, 4)})

    return pd.DataFrame(rows)
