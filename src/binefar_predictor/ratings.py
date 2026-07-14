"""Team-strength rating models.

Two engines:

* :class:`EloModel` — a fast scalar-rating baseline with margin-of-victory and
  home advantage. Used as a sanity check and for a quick strength ordering.
* :class:`DixonColesModel` — the workhorse. A time-weighted, L2-regularized
  Dixon-Coles bivariate-goals model (Dixon & Coles, 1997) giving per-team
  attack/defense parameters plus a home advantage and the low-score
  dependence term ``rho``. It yields full scoreline distributions, which the
  Monte-Carlo simulator samples from.

Only match results are required (goals + date + teams) — appropriate for a
tier-5 league where no xG or player-level data exists.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

LEAGUE_AVG = 0.0  # attack/defense of an unknown (new) team = league average


# =========================================================================== #
# Elo
# =========================================================================== #
@dataclass
class EloModel:
    """Standard Elo with margin-of-victory scaling and home advantage."""

    k: float = 24.0
    home_adv: float = 60.0
    base_rating: float = 1500.0
    ratings: dict[str, float] = field(default_factory=dict)

    def _r(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    @staticmethod
    def _expected(dr: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

    @staticmethod
    def _mov_multiplier(goal_diff: int, dr_winner: float) -> float:
        # FiveThirtyEight-style: dampen blowouts, correct for favourite inflation.
        gd = abs(goal_diff)
        if gd == 0:
            gd = 1
        return math.log(gd + 1) * (2.2 / (0.001 * dr_winner + 2.2))

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        """Process matches chronologically, updating ratings in place."""
        self.ratings = {}
        for row in matches.sort_values("timestamp").itertuples():
            h, a = row.home, row.away
            rh, ra = self._r(h), self._r(a)
            dr = (rh + self.home_adv) - ra
            e_home = self._expected(dr)
            gd = row.home_goals - row.away_goals
            s_home = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
            # rating diff from the winner's perspective, for MOV correction
            dr_winner = dr if gd > 0 else -dr
            g = self._mov_multiplier(gd, dr_winner) if gd != 0 else 1.0
            delta = self.k * g * (s_home - e_home)
            self.ratings[h] = rh + delta
            self.ratings[a] = ra - delta
        return self

    def win_probabilities(self, home: str, away: str) -> tuple[float, float, float]:
        """(home, draw, away) probabilities via an Elo->W/D/L mapping."""
        dr = (self._r(home) + self.home_adv) - self._r(away)
        e_home = self._expected(dr)
        # Empirical draw model: draws peak for even games, decay with |dr|.
        p_draw = 0.28 * math.exp(-abs(dr) / 220.0)
        p_home = e_home * (1 - p_draw)
        p_away = (1 - e_home) * (1 - p_draw)
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total


# =========================================================================== #
# Dixon-Coles
# =========================================================================== #
def _dc_tau(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> np.ndarray:
    """Low-score dependence correction applied to the (0/1 x 0/1) cells.

    ``lam`` and ``mu`` are per-match expected-goal arrays aligned with x, y.
    """
    tau = np.ones_like(x, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    return tau


@dataclass
class DixonColesModel:
    """Time-weighted, L2-regularized Dixon-Coles goals model.

    Parameters (packed into a single vector for the optimizer):
    ``attack[i]`` for each team, ``defense[i]`` for each team, global home
    advantage ``home``, intercept ``mu``, and low-score term ``rho``.
    Identifiability: ``sum(attack)=0`` and ``sum(defense)=0``.
    """

    half_life_days: float = 365.0
    l2: float = 0.05          # ridge shrinkage toward league average
    max_goals: int = 15
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home: float = 0.25
    mu: float = 0.0
    rho: float = -0.05
    _fit_result: object = None

    # -- fitting ------------------------------------------------------------ #
    def _time_weights(self, matches: pd.DataFrame, ref_ts: float) -> np.ndarray:
        if self.half_life_days <= 0:
            return np.ones(len(matches))
        xi = math.log(2) / (self.half_life_days * 86400.0)  # per second
        age = ref_ts - matches["timestamp"].to_numpy(dtype=float)
        age = np.clip(age, 0, None)
        return np.exp(-xi * age)

    def fit(self, matches: pd.DataFrame, ref_timestamp: float | None = None) -> "DixonColesModel":
        matches = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        self.teams = sorted(set(matches["home"]) | set(matches["away"]))
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}

        h_idx = matches["home"].map(idx).to_numpy()
        a_idx = matches["away"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy(dtype=int)
        ag = matches["away_goals"].to_numpy(dtype=int)

        ref_ts = ref_timestamp if ref_timestamp is not None else matches["timestamp"].max()
        weights = self._time_weights(matches, ref_ts)

        # param layout: [attack(n), defense(n), home, mu, rho]
        def unpack(p):
            return p[:n], p[n : 2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

        # precompute log-factorials for the observed goal counts
        log_fact_h = np.array([math.lgamma(g + 1) for g in hg])
        log_fact_a = np.array([math.lgamma(g + 1) for g in ag])

        def neg_log_like(p):
            attack, defense, home, mu, rho = unpack(p)
            log_lam = mu + home + attack[h_idx] + defense[a_idx]
            log_mu = mu + attack[a_idx] + defense[h_idx]
            lam = np.exp(log_lam)
            mu_ = np.exp(log_mu)
            # Poisson log-pmf for observed goals
            ll = hg * log_lam - lam - log_fact_h + ag * log_mu - mu_ - log_fact_a
            # Dixon-Coles low-score correction (only affects x,y in {0,1})
            tau = _dc_tau(hg, ag, lam, mu_, rho)
            tau = np.clip(tau, 1e-10, None)
            ll = ll + np.log(tau)
            nll = -np.sum(weights * ll)
            # L2 shrinkage toward league average (0)
            nll += self.l2 * (np.sum(attack ** 2) + np.sum(defense ** 2))
            return nll

        x0 = np.concatenate(
            [np.zeros(n), np.zeros(n), [0.25], [float(np.log(hg.mean() + 1e-6))], [-0.05]]
        )
        constraints = [
            {"type": "eq", "fun": lambda p: np.sum(p[:n])},          # sum attack = 0
            {"type": "eq", "fun": lambda p: np.sum(p[n : 2 * n])},   # sum defense = 0
        ]
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-2, 2), (-0.2, 0.2)]
        with warnings.catch_warnings():
            # SLSQP occasionally probes just outside bounds then clips — benign.
            warnings.simplefilter("ignore", RuntimeWarning)
            res = minimize(
                neg_log_like,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 400, "ftol": 1e-7},
            )
        self._fit_result = res
        attack, defense, home, mu, rho = unpack(res.x)
        self.attack = dict(zip(self.teams, attack))
        self.defense = dict(zip(self.teams, defense))
        self.home = float(home)
        self.mu = float(mu)
        self.rho = float(rho)
        return self

    # -- prediction --------------------------------------------------------- #
    def _params(self, team: str) -> tuple[float, float]:
        return self.attack.get(team, LEAGUE_AVG), self.defense.get(team, LEAGUE_AVG)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        ah, dh = self._params(home)
        aa, da = self._params(away)
        lam = math.exp(self.mu + self.home + ah + da)
        mu_ = math.exp(self.mu + aa + dh)
        return lam, mu_

    def scoreline_matrix(self, home: str, away: str) -> np.ndarray:
        """Full P(home=x, away=y) matrix, DC-corrected and normalized."""
        lam, mu_ = self.expected_goals(home, away)
        gm = self.max_goals
        xs = np.arange(gm + 1)
        p_home = poisson.pmf(xs, lam)
        p_away = poisson.pmf(xs, mu_)
        mat = np.outer(p_home, p_away)
        # DC correction on the four low-score cells
        mat[0, 0] *= 1 - lam * mu_ * self.rho
        mat[0, 1] *= 1 + lam * self.rho
        mat[1, 0] *= 1 + mu_ * self.rho
        mat[1, 1] *= 1 - self.rho
        mat = np.clip(mat, 0, None)
        return mat / mat.sum()

    def win_probabilities(self, home: str, away: str) -> tuple[float, float, float]:
        mat = self.scoreline_matrix(home, away)
        p_home = np.tril(mat, -1).sum()   # home_goals > away_goals
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        return float(p_home), float(p_draw), float(p_away)

    def strength_table(self) -> pd.DataFrame:
        """Per-team attack/defense and a net-strength ordering."""
        rows = []
        for t in self.teams:
            a, d = self.attack[t], self.defense[t]
            rows.append({"team": t, "attack": a, "defense": d, "net_strength": a - d})
        return (
            pd.DataFrame(rows)
            .sort_values("net_strength", ascending=False)
            .reset_index(drop=True)
        )
