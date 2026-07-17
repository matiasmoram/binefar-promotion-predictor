"""Model ensemble and cross-model agreement.

A single model can be quietly wrong, so we run the promotion simulation under
several rating variants and report the spread as an honest uncertainty band:

* Dixon-Coles, 1-year half-life (primary)
* Dixon-Coles, 6-month half-life (form-weighted)
* Dixon-Coles, 2-year half-life (long memory)
* Independent double-Poisson (rho fixed to 0 — tests the draw-correction's effect)

We also cross-check the *ordering* of team strength across three independent
rating systems (Dixon-Coles, Elo, pi-ratings) via Spearman correlation — if they
disagree wildly on who is strong, the promotion number is not trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .ratings import BivariatePoissonModel, DixonColesModel, EloModel, PiRatingModel
from .simulate import SeasonSimulator


@dataclass
class EnsembleResult:
    members: dict[str, float]        # member name -> promotion probability
    weights: dict[str, float]        # member name -> ensemble weight (sums to 1)
    mean: float                      # log-loss-weighted mean promotion probability
    min: float
    max: float
    std: float
    strength_agreement: dict[str, float]  # pairwise Spearman of team strengths

    def summary(self) -> str:
        lines = [f"Ensemble promotion probability: {self.mean:.1%} "
                 f"(range {self.min:.1%}–{self.max:.1%}, sd {self.std:.1%}; "
                 f"log-loss-weighted)"]
        for name, p in self.members.items():
            lines.append(f"  · {name:24s} {p:5.1%}  (w={self.weights.get(name, 0):.2f})")
        lines.append("  strength-ordering agreement (Spearman):")
        for pair, rho in self.strength_agreement.items():
            lines.append(f"    {pair:20s} {rho:+.2f}")
        return "\n".join(lines)


def _make_model(name: str):
    """Fresh, unfitted model for an ensemble member name."""
    return {
        "Dixon-Coles (hl=365d)": lambda: DixonColesModel(half_life_days=365, l2=0.05),
        "Dixon-Coles (hl=180d)": lambda: DixonColesModel(half_life_days=180, l2=0.05),
        "Dixon-Coles (hl=730d)": lambda: DixonColesModel(half_life_days=730, l2=0.05),
        "Independent Poisson": lambda: DixonColesModel(half_life_days=365, l2=0.05, fix_rho=0.0),
        "Bivariate Poisson": lambda: BivariatePoissonModel(half_life_days=365, l2=0.05),
    }[name]()


_MEMBER_NAMES = ["Dixon-Coles (hl=365d)", "Dixon-Coles (hl=180d)",
                 "Dixon-Coles (hl=730d)", "Independent Poisson", "Bivariate Poisson"]


def _holdout_logloss(name: str, matches: pd.DataFrame) -> float:
    """Out-of-sample W/D/L log-loss for a member: fit on all seasons but the
    last, score the last season. Lower = better; used for ensemble weighting."""
    last = matches.sort_values("timestamp")["season"].iloc[-1]
    start = matches[matches["season"] == last]["timestamp"].min()
    train = matches[matches["timestamp"] < start]
    test = matches[matches["season"] == last]
    if len(train) < 300 or test.empty:
        return 1.0986  # ln(3): a uniform-guess fallback
    model = _make_model(name).fit(train, ref_timestamp=start)
    ll, n = 0.0, 0
    for r in test.itertuples():
        if r.home not in model.attack or r.away not in model.attack:
            continue
        ph, pdr, pa = model.win_probabilities(r.home, r.away)
        gd = r.home_goals - r.away_goals
        p = ph if gd > 0 else (pdr if gd == 0 else pa)
        ll += -np.log(min(max(p, 1e-15), 1.0))
        n += 1
    return ll / n if n else 1.0986


def bootstrap_promotion(
    matches: pd.DataFrame,
    group: list[str],
    target: str = config.CLUB_NAME,
    n_boot: int = 40,
    n_sims: int = 3_000,
    half_life_days: float = 365.0,
    l2: float = 0.05,
    seed: int = 99,
) -> dict:
    """Promotion probability with **parameter uncertainty** via the bootstrap.

    A plain Monte-Carlo sim from point-estimate ratings captures only outcome
    randomness, not the fact that the ratings are themselves estimated from a
    finite (small, at tier 5) sample — so it is over-confident. Here we resample
    matches with replacement, refit, and re-simulate ``n_boot`` times; the spread
    of the resulting promotion probabilities is an honest estimate of that
    parameter uncertainty. Returns mean and a 90% interval.
    """
    rng = np.random.default_rng(seed)
    probs = []
    n = len(matches)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = matches.iloc[idx].reset_index(drop=True)
        model = DixonColesModel(half_life_days=half_life_days, l2=l2).fit(boot)
        res = SeasonSimulator(model, group, target=target, seed=int(rng.integers(1, 1_000_000))).run(n_sims=n_sims)
        probs.append(res.p_promotion)
    probs = np.array(probs)
    return {
        "n_boot": n_boot,
        "mean": round(float(probs.mean()), 4),
        "ci90": [round(float(np.percentile(probs, 5)), 4),
                 round(float(np.percentile(probs, 95)), 4)],
        "std": round(float(probs.std()), 4),
    }


def run_ensemble(
    matches: pd.DataFrame,
    group: list[str],
    target: str = config.CLUB_NAME,
    n_sims: int = 25_000,
) -> EnsembleResult:
    members: dict[str, float] = {}
    loglosses: dict[str, float] = {}
    primary_dc = None
    for name in _MEMBER_NAMES:
        model = _make_model(name).fit(matches)
        if primary_dc is None:
            primary_dc = model
        res = SeasonSimulator(model, group, target=target).run(n_sims=n_sims)
        members[name] = res.p_promotion
        loglosses[name] = _holdout_logloss(name, matches)

    probs = np.array(list(members.values()))
    # weight members by out-of-sample skill: softmax(-logloss / tau)
    ll = np.array([loglosses[n] for n in members])
    w = np.exp(-(ll - ll.min()) / 0.02)
    w = w / w.sum()
    weights = {n: float(wi) for n, wi in zip(members, w)}
    weighted_mean = float(np.sum(probs * w))

    # strength-ordering agreement across three independent rating systems
    elo = EloModel().fit(matches)
    pi = PiRatingModel().fit(matches)
    common = [t for t in group if t in primary_dc.attack and t in elo.ratings
              and (t in pi.home_r or t in pi.away_r)]
    dc_s = [primary_dc.attack[t] - primary_dc.defense[t] for t in common]
    elo_s = [elo.ratings[t] for t in common]
    pi_s = [pi.strength(t) for t in common]
    agreement = {}
    if len(common) >= 4:
        agreement["Dixon-Coles vs Elo"] = round(float(spearmanr(dc_s, elo_s).correlation), 3)
        agreement["Dixon-Coles vs pi"] = round(float(spearmanr(dc_s, pi_s).correlation), 3)
        agreement["Elo vs pi"] = round(float(spearmanr(elo_s, pi_s).correlation), 3)

    return EnsembleResult(
        members=members,
        weights=weights,
        mean=weighted_mean,
        min=float(probs.min()),
        max=float(probs.max()),
        std=float(probs.std()),
        strength_agreement=agreement,
    )
