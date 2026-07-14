"""Goalscorer prediction (Dirichlet-multinomial allocation).

The decomposition, forced by tier-5 data (no minutes/line-ups/xG):

    E[player goals] = E[team goals] (from the season simulator)
                       × player's share of team goals (this module)

**Data note.** Sofascore records every goal event (team totals are exact) but
names the scorer on only ~26% of goals at this tier, so incident-derived player
tallies badly undercount. We therefore use **Futbolme's season goleadores table**
(authoritative per-player season totals for the whole group) as the goal counts,
and use Sofascore incidents only to detect the penalty taker and own goals.

Model: a team's goals are split among players by a **Dirichlet-multinomial**
(overdispersed multinomial — scoring shares vary more than multinomial and stars
streak). Each player's Dirichlet prior mean is a positional base share; the
posterior blends prior with authoritative goals (empirical-Bayes shrinkage),
which handles small samples, **new signings**, and squad churn. Penalties are a
separate stream routed to the designated taker.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Relative scoring propensity by (normalized) position — Dirichlet prior mean and
# shrinkage target (research: lead striker ~25-35% of team goals, midfielders ~3-7%).
POSITION_PROPENSITY = {
    "centre-forward": 1.00, "second striker": 0.80, "forward": 0.90,
    "left winger": 0.65, "right winger": 0.65, "winger": 0.65,
    "attacking midfield": 0.55, "midfielder": 0.35, "central midfield": 0.30,
    "defensive midfield": 0.18, "left-back": 0.12, "right-back": 0.12,
    "centre-back": 0.14, "defender": 0.13, "goalkeeper": 0.01,
}
DEFAULT_PROPENSITY = 0.30
ALPHA0 = 25.0            # Dirichlet concentration = prior "team-goals of evidence"
RESIDUAL_SHARE = 0.04    # own goals + fringe/unknown scorers


def _norm(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _position_prior(position: str | None) -> float:
    if not position:
        return DEFAULT_PROPENSITY
    return POSITION_PROPENSITY.get(position.strip().lower(), DEFAULT_PROPENSITY)


def _name_matches(futbolme_name: str, squad_name: str) -> bool:
    """True if a Futbolme short name plausibly refers to a squad player.

    Futbolme uses a surname or nickname ("De Mesa", "Chicho", "Sanz"); TM uses
    the full name. We require the Futbolme name's tokens to be a *subset* of the
    squad name's tokens (so "de mesa" ⊆ "adria de mesa"), or a clean substring.
    A single short first-name token like "alex" is NOT enough on its own — it is
    only accepted inside a team where names are unambiguous (see caller).
    """
    f = set(_norm(futbolme_name).split())
    s = set(_norm(squad_name).split())
    if not f:
        return False
    # token-subset only — a raw substring test would mis-match "Sanz" ⊂ "Sanzol".
    return f <= s


def match_goals_to_squad(
    squad_names: list[str],
    team_goleadores: pd.DataFrame,
    league_goleadores: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Map each squad player to authoritative season goals.

    Two stages, to correctly handle both returning players and **new signings**:

    1. Match within the club's own goleadores rows (``team_goleadores``) — names
       are unambiguous within one squad, so token-subset matching is safe.
    2. For squad players still unmatched (potential signings from elsewhere),
       search the whole-league table (``league_goleadores``) but require a
       *surname* match (the player's last name equals a Futbolme name token of
       length >= 4) to avoid first-name collisions like "Álex".
    """
    goals: dict[str, int] = {name: 0 for name in squad_names}
    matched: set[str] = set()

    # Stage 1: within-team (safe token-subset matching)
    for _, row in team_goleadores.iterrows():
        for sname in squad_names:
            if sname in matched:
                continue
            if _name_matches(row["player"], sname):
                goals[sname] = max(goals[sname], int(row["goals"]))
                matched.add(sname)
                break

    # Stage 2: league-wide surname match for the unmatched (signings)
    if league_goleadores is not None:
        for sname in squad_names:
            if sname in matched:
                continue
            surname = _norm(sname).split()[-1] if _norm(sname) else ""
            if len(surname) < 4:
                continue
            cand = league_goleadores[
                league_goleadores["player"].apply(
                    lambda p: surname in _norm(p).split()
                )
            ]
            # only assign when the surname is unambiguous league-wide; a
            # ".max() across namesakes" would fabricate goals for a signing.
            if len(cand) == 1:
                goals[sname] = int(cand["goals"].iloc[0])
                matched.add(sname)
    return goals


# =========================================================================== #
# Target-club model
# =========================================================================== #
@dataclass
class GoalscorerModel:
    squad: pd.DataFrame                   # columns: name, position (26/27 squad)
    player_goals: dict[str, int]          # authoritative goals per squad player
    penalty_taker: str | None = None
    penalty_fraction: float = 0.10
    own_goals: int = 0
    alpha0: float = ALPHA0
    players: list[str] = field(default_factory=list)
    alpha_post: np.ndarray = None
    positions: dict[str, str] = field(default_factory=dict)
    is_new: dict[str, bool] = field(default_factory=dict)

    def fit(self) -> "GoalscorerModel":
        squad = [(r["name"], r.get("position")) for _, r in self.squad.iterrows()]
        propensity = np.array([_position_prior(pos) for _, pos in squad])
        prior_share = propensity / propensity.sum() * (1 - RESIDUAL_SHARE)

        # The authoritative goal totals INCLUDE penalties. Penalties are re-added
        # as a separate additive stream in allocate(), so we must remove the
        # taker's expected penalties from the open-play Dirichlet base to avoid
        # double-counting them.
        team_total = sum(int(v) for v in self.player_goals.values()) or 1
        taker_pens = round(self.penalty_fraction * team_total)
        taker_norm = _norm(self.penalty_taker) if self.penalty_taker else ""
        taker_tokens = set(taker_norm.split())

        players, alpha_post, positions, is_new = [], [], {}, {}
        for (name, pos), pshare in zip(squad, prior_share):
            g = int(self.player_goals.get(name, 0))
            open_g = g
            is_taker = taker_norm and (
                taker_norm in _norm(name) or (taker_tokens and taker_tokens <= set(_norm(name).split()))
            )
            if is_taker:
                open_g = max(0, g - taker_pens)  # strip penalties from the base
            alpha_post.append(self.alpha0 * pshare + open_g)
            players.append(name)
            positions[name] = pos
            is_new[name] = g == 0  # no goal record for this club last season
        # residual bucket
        players.append("(other/own goals)")
        alpha_post.append(self.alpha0 * RESIDUAL_SHARE + self.own_goals)
        positions["(other/own goals)"] = None
        is_new["(other/own goals)"] = False

        self.players = players
        self.alpha_post = np.array(alpha_post, dtype=float)
        self.positions = positions
        self.is_new = is_new
        return self

    def _taker_index(self) -> int | None:
        """Index of the penalty taker in ``self.players`` (accent-insensitive)."""
        if not self.penalty_taker:
            return None
        tk = _norm(self.penalty_taker)
        tk_tokens = set(tk.split())
        for i, pl in enumerate(self.players):
            pn = _norm(pl)
            if pn == tk or tk in pn or (tk_tokens and tk_tokens <= set(pn.split())):
                return i
        return None

    def allocate(self, team_goal_samples: np.ndarray, seed: int = 123,
                 max_sims: int = 30_000) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        samples = np.asarray(team_goal_samples)
        if len(samples) > max_sims:
            samples = rng.choice(samples, size=max_sims, replace=False)
        n_sims = len(samples)
        K = len(self.players)
        taker_idx = self._taker_index()

        goals_matrix = np.zeros((K, n_sims), dtype=np.int32)
        W = rng.dirichlet(self.alpha_post, size=n_sims)
        for s in range(n_sims):
            total = int(samples[s])
            if total <= 0:
                continue
            n_pen = rng.binomial(total, self.penalty_fraction) if taker_idx is not None else 0
            n_open = total - n_pen
            if n_open > 0:
                goals_matrix[:, s] += rng.multinomial(n_open, W[s])
            if n_pen and taker_idx is not None:
                goals_matrix[taker_idx, s] += n_pen

        exp_goals = goals_matrix.mean(axis=1)
        p10 = np.percentile(goals_matrix, 10, axis=1)
        p90 = np.percentile(goals_matrix, 90, axis=1)
        # top-scorer race is among named players — the residual bucket (last row,
        # own goals / unknown) can never be the identified top scorer.
        race = goals_matrix.astype(float) + rng.random(goals_matrix.shape) * 0.01
        race[K - 1] = -1.0  # residual bucket masked out of the argmax
        top_prob = np.bincount(np.argmax(race, axis=0), minlength=K) / n_sims

        rows = []
        for i, pl in enumerate(self.players):
            rows.append({
                "player": pl,
                "position": self.positions.get(pl),
                "goals_last_season": int(self.player_goals.get(pl, 0)),
                "new_or_scoreless": bool(self.is_new.get(pl, False)),
                "is_pen_taker": i == taker_idx,
                "exp_goals": round(float(exp_goals[i]), 2),
                "p10": int(p10[i]),
                "p90": int(p90[i]),
                "p_team_top_scorer": round(float(top_prob[i]), 3),
            })
        return (pd.DataFrame(rows)
                .sort_values("exp_goals", ascending=False)
                .reset_index(drop=True))

    def anytime_scorer_table(self, mean_team_goals_per_match: float) -> pd.DataFrame:
        shares = self.alpha_post / self.alpha_post.sum()
        rows = [{"player": pl,
                 "p_anytime": round(1 - float(np.exp(-w * mean_team_goals_per_match)), 3)}
                for pl, w in zip(self.players, shares)]
        return (pd.DataFrame(rows)
                .sort_values("p_anytime", ascending=False).reset_index(drop=True))


# =========================================================================== #
# League-wide pichichi race (uses authoritative goleadores table)
# =========================================================================== #
def league_top_scorer(
    all_goals_for: np.ndarray,
    teams: list[str],
    goleadores: pd.DataFrame,
    seed: int = 321,
    max_sims: int = 20_000,
    top_n: int = 20,
    alpha0: float = ALPHA0,
) -> pd.DataFrame:
    """Pichichi probabilities across the group, from authoritative goal totals.

    ``all_goals_for`` (n_teams, n_sims) from the simulator; ``goleadores`` the
    Futbolme table (player, team, goals). Team-name matching between the two
    sources is by case-insensitive substring on a shared token.
    """
    rng = np.random.default_rng(seed)
    n_teams, n_sims = all_goals_for.shape
    if n_sims > max_sims:
        idx = rng.choice(n_sims, size=max_sims, replace=False)
        all_goals_for = all_goals_for[:, idx]
        n_sims = max_sims

    # Generic club words that must not drive team matching (else "Atlético
    # Calatayud" would match "Atlético Monzón" on "atletico").
    STOP = {"cd", "cf", "ca", "sd", "ud", "ad", "b", "atletico", "athletic",
            "real", "deportivo", "union", "sociedad", "club", "de", "el", "la",
            "rz", "cdj"}

    def _key(name: str) -> set[str]:
        return set(_norm(name).split()) - STOP

    def match_team(sim_team: str) -> pd.DataFrame:
        toks = _key(sim_team)
        m = goleadores[goleadores["team"].apply(lambda t: bool(toks & _key(t)))]
        return m

    global_players: list[str] = []
    team_slices: dict[int, tuple[list[int], np.ndarray]] = {}
    for ti, t in enumerate(teams):
        gt = match_team(t)
        names = list(gt["player"]) + [f"__other__{t}"]
        counts = np.append(gt["goals"].to_numpy(dtype=float), 0.0)
        prior = np.full(len(names), alpha0 / max(1, len(names)))
        alpha = prior + counts
        base = list(range(len(global_players), len(global_players) + len(names)))
        global_players.extend(f"{n} ({t})" for n in names)
        team_slices[ti] = (base, alpha)

    season_goals = np.zeros((len(global_players), n_sims), dtype=np.int32)
    for ti in range(n_teams):
        base, alpha = team_slices[ti]
        W = rng.dirichlet(alpha, size=n_sims)
        totals = all_goals_for[ti]
        for s in range(n_sims):
            tot = int(totals[s])
            if tot > 0:
                season_goals[base, s] += rng.multinomial(tot, W[s])

    # The top-scorer race is decided among *named* players only. Residual
    # "__other__" buckets (unnamed goals; whole newcomer teams with no goleadores
    # data) cannot be the identified pichichi, so we mask them out of the argmax
    # rather than let a diffuse bucket win the crown.
    is_real = np.array([not name.startswith("__other__") for name in global_players])
    real_goals = season_goals.copy().astype(float)
    real_goals[~is_real] = -1.0  # never win the argmax
    noisy = real_goals + rng.random(real_goals.shape) * 0.01
    top_prob = np.bincount(np.argmax(noisy, axis=0), minlength=len(global_players)) / n_sims
    exp_goals = season_goals.mean(axis=1)
    rows = [{"player_team": name, "exp_goals": round(float(exp_goals[i]), 2),
             "p_pichichi": round(float(top_prob[i]), 3)}
            for i, name in enumerate(global_players) if is_real[i]]
    return (pd.DataFrame(rows).sort_values("p_pichichi", ascending=False)
            .head(top_n).reset_index(drop=True))
