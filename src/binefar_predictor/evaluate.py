"""Out-of-sample validation of the ratings model.

Two complementary backtests, both strictly walk-forward (train only on the past
— no leakage):

* :func:`match_backtest` — the data-rich one. For every match after a warm-up
  period, predict W/D/L from a model fit only on earlier matches, then score the
  probabilities with log-loss, the (multiclass) Brier score and the Ranked
  Probability Score, against naive baselines. Also returns calibration-curve
  data (predicted vs. observed frequency).

* :func:`champion_backtest` — the headline sanity check. For each completed
  season, fit on all prior matches and record the model-implied probability that
  the eventual champion would win the title.

With only ~9 promotion events, match-level scoring is where statistical power
lives; the champion backtest is a reality check, not a significance test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .ratings import DixonColesModel


def _clip(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(p, eps, 1 - eps)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass log-loss. probs (n,3) for [home, draw, away]; outcomes in 0/1/2."""
    p = _clip(probs)
    return float(-np.mean(np.log(p[np.arange(len(outcomes)), outcomes])))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score for ordinal outcome home(0)>draw(1)>away(2)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    # average over the (k-1) thresholds
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1)[:, None] / (probs.shape[1] - 1)))


@dataclass
class BacktestReport:
    n_matches: int
    log_loss: float
    brier: float
    rps: float
    baseline_log_loss: float
    baseline_brier: float
    accuracy: float
    calibration: dict = field(default_factory=dict)  # bin_mid -> observed freq

    def summary(self) -> str:
        return (
            f"Walk-forward match backtest ({self.n_matches:,} matches)\n"
            f"  log-loss:  {self.log_loss:.4f}  (baseline {self.baseline_log_loss:.4f})\n"
            f"  Brier:     {self.brier:.4f}  (baseline {self.baseline_brier:.4f})\n"
            f"  RPS:       {self.rps:.4f}\n"
            f"  top-pick accuracy: {self.accuracy:.1%}"
        )


def match_backtest(
    matches: pd.DataFrame,
    half_life_days: float = 365.0,
    l2: float = 0.05,
    min_train_matches: int = 400,
    refit_every: int = 40,
) -> BacktestReport:
    """Predict each match from a model fit only on strictly earlier matches.

    To keep it tractable the model is refit every ``refit_every`` matches
    (ratings drift slowly, so this is a fine approximation of a per-match refit).
    """
    matches = matches.sort_values("timestamp").reset_index(drop=True)
    probs_list: list[list[float]] = []
    outcomes: list[int] = []

    model: DixonColesModel | None = None
    last_fit_at = -(10**9)

    for i in range(min_train_matches, len(matches)):
        if model is None or (i - last_fit_at) >= refit_every:
            train = matches.iloc[:i]
            model = DixonColesModel(half_life_days=half_life_days, l2=l2).fit(
                train, ref_timestamp=matches.iloc[i]["timestamp"]
            )
            last_fit_at = i
        row = matches.iloc[i]
        # unknown team in the future match -> skip (no fair prediction)
        if row.home not in model.attack or row.away not in model.attack:
            continue
        ph, pdr, pa = model.win_probabilities(row.home, row.away)
        probs_list.append([ph, pdr, pa])
        gd = row.home_goals - row.away_goals
        outcomes.append(0 if gd > 0 else (1 if gd == 0 else 2))

    probs = np.array(probs_list)
    outc = np.array(outcomes)

    # baseline: fixed historical base rates for home/draw/away
    base_rates = np.bincount(outc, minlength=3) / len(outc)
    base_probs = np.tile(base_rates, (len(outc), 1))

    # calibration: bin the predicted P(home win) vs realized home-win rate
    calib = {}
    home_p = probs[:, 0]
    home_win = (outc == 0).astype(float)
    bins = np.linspace(0, 1, 11)
    which = np.digitize(home_p, bins) - 1
    for b in range(10):
        mask = which == b
        if mask.sum() >= 20:
            calib[round(float(np.mean(home_p[mask])), 3)] = round(
                float(np.mean(home_win[mask])), 3
            )

    return BacktestReport(
        n_matches=len(outc),
        log_loss=log_loss(probs, outc),
        brier=brier(probs, outc),
        rps=rps(probs, outc),
        baseline_log_loss=log_loss(base_probs, outc),
        baseline_brier=brier(base_probs, outc),
        accuracy=float(np.mean(np.argmax(probs, axis=1) == outc)),
        calibration=calib,
    )


def champion_backtest(
    matches: pd.DataFrame,
    standings_by_season: dict[str, pd.DataFrame],
    half_life_days: float = 365.0,
    l2: float = 0.05,
    n_sims: int = 5_000,
) -> pd.DataFrame:
    """Pre-season title probability assigned to the eventual champion.

    For each season with a known final table, fit on matches before that season
    began and simulate; report the probability mass the model placed on the team
    that actually won.
    """
    from .simulate import SeasonSimulator

    matches = matches.sort_values("timestamp")
    rows = []
    for label, table in standings_by_season.items():
        season_matches = matches[matches["season"] == label]
        if season_matches.empty:
            continue
        start_ts = season_matches["timestamp"].min()
        train = matches[matches["timestamp"] < start_ts]
        if len(train) < 300:
            continue
        model = DixonColesModel(half_life_days=half_life_days, l2=l2).fit(
            train, ref_timestamp=start_ts
        )
        teams = list(table["team"])
        # need the champion in the group and at least some rated teams
        champion = table.sort_values("position").iloc[0]["team"]
        rated = sum(t in model.attack for t in teams)
        if rated < len(teams) // 2 or champion not in model.attack:
            continue
        sim = SeasonSimulator(model, teams, target=champion, seed=7)
        res = sim.run(n_sims=n_sims)
        rows.append(
            {
                "season": label,
                "actual_champion": champion,
                "model_p_direct": round(res.p_direct, 3),
                "model_mean_pos": round(res.mean_position, 2),
            }
        )
    return pd.DataFrame(rows)
