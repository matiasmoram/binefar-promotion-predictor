"""Monte-Carlo season simulation and promotion probability.

Given a fitted :class:`~binefar_predictor.ratings.DixonColesModel` and the list
of teams in the target season, we:

1. build the full double round-robin fixture list;
2. precompute each fixture's scoreline distribution from the model;
3. simulate the season ``n_sims`` times (sampling scorelines, awarding points,
   ranking by points -> goal difference -> goals for);
4. resolve promotion for the target club: champion => direct promotion; a
   top-5 finish => territorial play-off (simulated) whose winner converts the
   national phase with probability :data:`config.NATIONAL_PHASE_CONVERSION`.

The result is an empirical promotion probability with a Monte-Carlo standard
error, plus finishing-position and points distributions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .ratings import DixonColesModel


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def double_round_robin(teams: list[str]) -> list[tuple[str, str]]:
    """Every ordered pair (home, away) — each team hosts every other once."""
    return [(h, a) for h in teams for a in teams if h != a]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    team: str
    n_sims: int
    p_promotion: float
    p_direct: float           # champion
    p_playoff_reached: float  # finished 2nd-5th
    p_playoff_won: float      # won the territorial play-off
    se: float                 # Monte-Carlo standard error of p_promotion
    position_dist: np.ndarray  # P(finish in position k), index 0 -> 1st
    mean_points: float
    points_pctiles: dict[str, float]
    mean_position: float
    teams: list[str]
    target_goals_for: np.ndarray = None  # (n_sims,) simulated goals scored
    all_goals_for: np.ndarray = None     # (n_teams, n_sims) goals for every team
    league_table: list = None            # projected table: per-team aggregates

    def summary(self) -> str:
        lines = [
            f"Promotion probability for {self.team}: {self.p_promotion:.1%} "
            f"(±{self.se:.1%}, {self.n_sims:,} sims)",
            f"  · direct (champion):        {self.p_direct:.1%}",
            f"  · reached play-off (2-5):   {self.p_playoff_reached:.1%}",
            f"  · won territorial play-off: {self.p_playoff_won:.1%}",
            f"  · mean finishing position:  {self.mean_position:.1f}",
            f"  · mean points:              {self.mean_points:.1f} "
            f"(p10={self.points_pctiles['p10']:.0f}, "
            f"p50={self.points_pctiles['p50']:.0f}, "
            f"p90={self.points_pctiles['p90']:.0f})",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class SeasonSimulator:
    def __init__(
        self,
        model: DixonColesModel,
        teams: list[str],
        target: str = config.CLUB_NAME,
        newcomer_penalty: float = config.NEWCOMER_NET_STRENGTH_PENALTY,
        national_conversion: float = config.NATIONAL_PHASE_CONVERSION,
        seed: int = 20262027,
    ) -> None:
        self.model = model
        self.teams = list(teams)
        self.n = len(self.teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        if target not in self.idx:
            raise ValueError(
                f"Target team {target!r} is not in the target-season group."
            )
        self.team_name = target
        self.newcomer_penalty = newcomer_penalty
        self.national_conversion = national_conversion
        self.rng = np.random.default_rng(seed)

        # Assign attack/defense for every team, with a cold-start rule for
        # teams the model never saw (newcomers): league average minus penalty,
        # split evenly across attack and defense.
        self._attack = {}
        self._defense = {}
        for t in self.teams:
            if t in self.model.attack:
                self._attack[t] = self.model.attack[t]
                self._defense[t] = self.model.defense[t]
            else:
                self._attack[t] = -newcomer_penalty / 2.0
                self._defense[t] = +newcomer_penalty / 2.0

        self.fixtures = double_round_robin(self.teams)

    # -- scoreline model with cold-start override -------------------------- #
    def _expected_goals(self, home: str, away: str) -> tuple[float, float]:
        import math

        lam = math.exp(
            self.model.mu + self.model.home + self._attack[home] + self._defense[away]
        )
        mu_ = math.exp(self.model.mu + self._attack[away] + self._defense[home])
        return lam, mu_

    def _scoreline_pmf_flat(self, home: str, away: str) -> np.ndarray:
        from scipy.stats import poisson

        lam, mu_ = self._expected_goals(home, away)
        gm = self.model.max_goals
        xs = np.arange(gm + 1)
        mat = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu_))
        mat[0, 0] *= 1 - lam * mu_ * self.model.rho
        mat[0, 1] *= 1 + lam * self.model.rho
        mat[1, 0] *= 1 + mu_ * self.model.rho
        mat[1, 1] *= 1 - self.model.rho
        mat = np.clip(mat, 0, None)
        return (mat / mat.sum()).ravel()

    def _pairwise_home_win(self, home: str, away: str) -> tuple[float, float, float]:
        """(home win, draw, away win) for a single play-off match."""
        from scipy.stats import poisson

        lam, mu_ = self._expected_goals(home, away)
        gm = self.model.max_goals
        xs = np.arange(gm + 1)
        mat = np.outer(poisson.pmf(xs, lam), poisson.pmf(xs, mu_))
        mat[0, 0] *= 1 - lam * mu_ * self.model.rho
        mat[0, 1] *= 1 + lam * self.model.rho
        mat[1, 0] *= 1 + mu_ * self.model.rho
        mat[1, 1] *= 1 - self.model.rho
        mat = np.clip(mat, 0, None)
        mat /= mat.sum()
        p_home = np.tril(mat, -1).sum()
        p_draw = np.trace(mat)
        return float(p_home), float(p_draw), float(1 - p_home - p_draw)

    # -- main loop --------------------------------------------------------- #
    def run(self, n_sims: int = 50_000) -> SimulationResult:
        gm1 = self.model.max_goals + 1
        n_fix = len(self.fixtures)

        # Sample all fixture scorelines: shape (n_fixtures, n_sims)
        home_goals = np.empty((n_fix, n_sims), dtype=np.int16)
        away_goals = np.empty((n_fix, n_sims), dtype=np.int16)
        outcomes = np.arange(gm1 * gm1)
        for f, (h, a) in enumerate(self.fixtures):
            pmf = self._scoreline_pmf_flat(h, a)
            draws = self.rng.choice(outcomes, size=n_sims, p=pmf)
            home_goals[f] = draws // gm1
            away_goals[f] = draws % gm1

        # Accumulate points / GD / GF per team across fixtures
        points = np.zeros((self.n, n_sims), dtype=np.int32)
        gf = np.zeros((self.n, n_sims), dtype=np.int32)
        ga = np.zeros((self.n, n_sims), dtype=np.int32)
        for f, (h, a) in enumerate(self.fixtures):
            hi, ai = self.idx[h], self.idx[a]
            hg, ag = home_goals[f], away_goals[f]
            home_win = hg > ag
            draw = hg == ag
            away_win = hg < ag
            points[hi] += 3 * home_win + 1 * draw
            points[ai] += 3 * away_win + 1 * draw
            gf[hi] += hg
            ga[hi] += ag
            gf[ai] += ag
            ga[ai] += hg
        gd = gf - ga

        # Rank each simulation: points -> GD -> GF, with tiny noise to break
        # remaining exact ties randomly (models the coin-flip nature of h2h).
        noise = self.rng.random((self.n, n_sims)) * 1e-3
        # Lexicographic key points -> GD -> GF. Spacing must dominate the next
        # level: GD term *1e3 > max GF (~150); points *1e6 > max GD term
        # ((gd+500)*1e3 <= ~6.5e5 < 1e6). Max points ~102 -> ~1.02e8, safe in f64.
        score = points * 1e6 + (gd + 500) * 1e3 + gf + noise
        # position: 1 = best. argsort descending along team axis.
        order = np.argsort(-score, axis=0)             # team indices best->worst
        rank = np.empty((self.n, n_sims), dtype=np.int16)
        rows = np.arange(self.n)[:, None]
        rank[order, np.broadcast_to(np.arange(n_sims), order.shape)] = (
            rows + 1
        )

        ti = self.idx[self.team_name]
        target_rank = rank[ti]                          # (n_sims,)

        # Projected full-league table: per-team aggregates across all sims.
        mean_pts = points.mean(axis=1)
        mean_pos = rank.mean(axis=1)
        p_champ = (rank == 1).mean(axis=1)
        p_top5 = (rank <= (1 + config.PLAYOFF_SLOTS)).mean(axis=1)
        league_table = sorted(
            (
                {
                    "team": self.teams[i],
                    "mean_points": round(float(mean_pts[i]), 1),
                    "mean_position": round(float(mean_pos[i]), 1),
                    "p_champion": round(float(p_champ[i]), 3),
                    "p_top5": round(float(p_top5[i]), 3),
                    "is_target": self.teams[i] == self.team_name,
                    # full finishing-position distribution (for the heatmap)
                    "pos_dist": [
                        round(float(x), 4)
                        for x in np.bincount(rank[i], minlength=self.n + 1)[1:self.n + 1]
                        / n_sims
                    ],
                }
                for i in range(self.n)
            ),
            key=lambda r: -r["mean_points"],
        )

        # -- promotion resolution ----------------------------------------- #
        promoted = np.zeros(n_sims, dtype=bool)
        is_champion = target_rank == 1
        promoted |= is_champion

        reached_playoff = (target_rank >= 2) & (
            target_rank <= 1 + config.PLAYOFF_SLOTS
        )
        won_playoff = np.zeros(n_sims, dtype=bool)

        # Simulate the territorial play-off only for sims where the target
        # reached it. Identify the four seeds (ranks 2..5) per such sim.
        po_sims = np.where(reached_playoff)[0]
        if po_sims.size:
            # seed -> team index, for ranks 2..5
            # order[k, s] gives the team index finishing in position k+1.
            seed_team = order[1:1 + config.PLAYOFF_SLOTS][:, po_sims]  # (4, m)
            won = self._simulate_playoffs(seed_team, ti)
            won_playoff[po_sims] = won

        # National-phase conversion for the territorial winner.
        conv = self.rng.random(n_sims) < self.national_conversion
        promoted |= won_playoff & conv

        # -- aggregate metrics -------------------------------------------- #
        p_prom = promoted.mean()
        se = float(np.sqrt(p_prom * (1 - p_prom) / n_sims))
        pos_dist = np.bincount(target_rank, minlength=self.n + 1)[1:] / n_sims
        target_points = points[ti]

        return SimulationResult(
            team=self.team_name,
            n_sims=n_sims,
            p_promotion=float(p_prom),
            p_direct=float(is_champion.mean()),
            p_playoff_reached=float(reached_playoff.mean()),
            p_playoff_won=float(won_playoff.mean()),
            se=se,
            position_dist=pos_dist,
            mean_points=float(target_points.mean()),
            points_pctiles={
                "p10": float(np.percentile(target_points, 10)),
                "p50": float(np.percentile(target_points, 50)),
                "p90": float(np.percentile(target_points, 90)),
            },
            mean_position=float(target_rank.mean()),
            teams=self.teams,
            target_goals_for=gf[ti].copy(),
            all_goals_for=gf.copy(),
            league_table=league_table,
        )

    def _simulate_playoffs(self, seed_team: np.ndarray, target_idx: int) -> np.ndarray:
        """Vectorized single-match bracket: (2v5, 3v4) semis then final.

        ``seed_team`` is shape (4, m): rows are seeds 2,3,4,5; higher seed hosts.
        Returns a boolean array (m,): did the target win the territorial title?
        """
        m = seed_team.shape[1]
        rng = self.rng
        team_list = self.teams

        # Precompute pairwise home-win prob lazily via cache.
        cache: dict[tuple[int, int], float] = {}

        def home_win_prob(hi: int, ai: int) -> float:
            key = (hi, ai)
            if key not in cache:
                ph, pd_, pa = self._pairwise_home_win(team_list[hi], team_list[ai])
                # single elimination: split the draw mass by relative strength
                cache[key] = ph + pd_ * (ph / (ph + pa + 1e-9))
            return cache[key]

        def play(h_idx_arr: np.ndarray, a_idx_arr: np.ndarray) -> np.ndarray:
            """Return winner team-index per sim; higher seed (h) hosts."""
            probs = np.array(
                [home_win_prob(int(h), int(a)) for h, a in zip(h_idx_arr, a_idx_arr)]
            )
            h_wins = rng.random(len(probs)) < probs
            return np.where(h_wins, h_idx_arr, a_idx_arr)

        s2, s3, s4, s5 = seed_team[0], seed_team[1], seed_team[2], seed_team[3]
        # Semifinals: 2 vs 5, 3 vs 4 (higher seed hosts)
        semi1 = play(s2, s5)
        semi2 = play(s3, s4)
        # Final: higher seed hosts. Determine which finalist is higher-seeded.
        # Build seed rank lookup per sim.
        # seed order is fixed (row0 best), so compare positions in seed_team.
        def higher_seed(a_idx: np.ndarray, b_idx: np.ndarray) -> np.ndarray:
            # position (row) of each finalist within seed_team columns
            # semi1 winner came from rows {0,3}; semi2 from rows {1,2}.
            # Compare by finding row index; lower row = higher seed.
            def row_of(win, r_hi, r_lo, hi_arr, lo_arr):
                return np.where(win == hi_arr, r_hi, r_lo)

            row_a = row_of(a_idx, 0, 3, s2, s5)
            row_b = row_of(b_idx, 1, 2, s3, s4)
            a_is_home = row_a < row_b
            home = np.where(a_is_home, a_idx, b_idx)
            away = np.where(a_is_home, b_idx, a_idx)
            return home, away

        fh, fa = higher_seed(semi1, semi2)
        champion = play(fh, fa)
        return champion == target_idx
