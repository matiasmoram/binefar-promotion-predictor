"""End-to-end orchestration: data -> ratings -> simulation -> report.

``run_prediction`` is the single entry point used by the CLI. It produces a
:class:`PredictionReport` (also serialized to ``models/prediction.json`` and
``models/report.md``) and, optionally, diagnostic plots under ``models/``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data
from .ratings import DixonColesModel, EloModel
from .simulate import SeasonSimulator, SimulationResult


@dataclass
class PredictionReport:
    club: str
    target_season: str
    generated_at: str
    n_matches_trained: int
    n_seasons: int
    promotion_probability: float
    p_direct: float
    p_playoff_reached: float
    p_playoff_won: float
    monte_carlo_se: float
    mean_position: float
    mean_points: float
    position_distribution: list[float]
    strength_rank_in_group: int
    group: list[str]
    model_params: dict
    squad: list[dict] = field(default_factory=list)
    backtest: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)

    def headline(self) -> str:
        return (
            f"{self.club} — probability of promotion in {self.target_season}: "
            f"{self.promotion_probability:.1%} (±{self.monte_carlo_se:.1%})"
        )


def _group_strength_rank(model: DixonColesModel, group: list[str], team: str) -> int:
    def net(t):
        return model.attack.get(t, 0.0) - model.defense.get(t, 0.0)

    ordered = sorted(group, key=net, reverse=True)
    return ordered.index(team) + 1 if team in ordered else -1


def run_prediction(
    target: str = config.CLUB_NAME,
    n_sims: int = 50_000,
    half_life_days: float = 365.0,
    l2: float = 0.05,
    prefer_snapshot: bool = False,
    force_refresh: bool = False,
    include_squad: bool = True,
    include_backtest: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> PredictionReport:
    def _say(*a):
        if verbose:
            print(*a)

    _say(f"[1/6] Loading data (prefer_snapshot={prefer_snapshot}) …")
    matches, latest = data.load(prefer_snapshot=prefer_snapshot, force_refresh=force_refresh)
    n_seasons = matches["season"].nunique() if "season" in matches else 0
    _say(f"      {len(matches):,} matches across {n_seasons} seasons.")

    _say("[2/6] Fitting Dixon-Coles ratings …")
    model = DixonColesModel(half_life_days=half_life_days, l2=l2).fit(matches)
    elo = EloModel().fit(matches)

    _say("[3/6] Projecting the target-season group …")
    group = data.project_target_group(latest)
    notes = []
    if target not in group:
        # target unexpectedly absent (e.g. relegated) — still simulate by force-adding
        group = group[:-1] + [target]
        notes.append(f"{target} was not in the auto-projected group; forced in.")

    _say(f"[4/6] Monte-Carlo simulating {n_sims:,} seasons …")
    sim = SeasonSimulator(model, group, target=target)
    result: SimulationResult = sim.run(n_sims=n_sims)
    _say("      " + result.summary().replace("\n", "\n      "))

    squad_records: list[dict] = []
    if include_squad:
        _say("[5/6] Fetching squad (Transfermarkt, best-effort) …")
        try:
            from . import transfermarkt as tm

            players = tm.fetch_squad(season_year=int("20" + config.TARGET_SEASON[:2]))
            if not players:
                players = tm.fetch_squad(season_year=2025)
            squad_records = [asdict(p) for p in players]
            val = tm.squad_value_summary(players)
            _say(f"      squad size {val['squad_size']}, "
                 f"valued players {val['n_valued']} "
                 f"(market values are typically empty at tier 5).")
            if val["n_valued"] == 0:
                notes.append(
                    "Transfermarkt lists no market values for this squad (normal "
                    "at tier 5), so no value-based strength prior was applied."
                )
        except Exception as exc:
            notes.append(f"Squad fetch failed: {exc}")
    else:
        _say("[5/6] Skipping squad fetch.")

    backtest_dict = None
    if include_backtest:
        _say("[6/6] Running walk-forward backtest …")
        from .evaluate import match_backtest

        rep = match_backtest(matches, half_life_days=half_life_days, l2=l2, refit_every=60)
        backtest_dict = {
            "n_matches": rep.n_matches,
            "log_loss": round(rep.log_loss, 4),
            "brier": round(rep.brier, 4),
            "rps": round(rep.rps, 4),
            "baseline_log_loss": round(rep.baseline_log_loss, 4),
            "baseline_brier": round(rep.baseline_brier, 4),
            "accuracy": round(rep.accuracy, 4),
            "calibration": rep.calibration,
        }
        _say("      " + rep.summary().replace("\n", "\n      "))
    else:
        _say("[6/6] Skipping backtest.")

    report = PredictionReport(
        club=target,
        target_season=config.TARGET_SEASON,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        n_matches_trained=len(matches),
        n_seasons=int(n_seasons),
        promotion_probability=result.p_promotion,
        p_direct=result.p_direct,
        p_playoff_reached=result.p_playoff_reached,
        p_playoff_won=result.p_playoff_won,
        monte_carlo_se=result.se,
        mean_position=result.mean_position,
        mean_points=result.mean_points,
        position_distribution=result.position_dist.tolist(),
        strength_rank_in_group=_group_strength_rank(model, group, target),
        group=group,
        model_params={
            "half_life_days": half_life_days,
            "l2": l2,
            "home_advantage": round(model.home, 4),
            "mu": round(model.mu, 4),
            "rho": round(model.rho, 4),
            "national_phase_conversion": config.NATIONAL_PHASE_CONVERSION,
            "newcomer_net_strength_penalty": config.NEWCOMER_NET_STRENGTH_PENALTY,
            "elo_binefar": round(elo.ratings.get(target, elo.base_rating), 1),
        },
        squad=squad_records,
        backtest=backtest_dict,
        notes=notes,
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_json(config.MODELS_DIR / "prediction.json")
    _write_markdown(report, model, result, config.MODELS_DIR / "report.md")
    if make_plots:
        _make_plots(report, model, result, group, config.MODELS_DIR)
    _say(f"\nSaved: {config.MODELS_DIR/'prediction.json'}, report.md"
         + (", plots" if make_plots else ""))
    return report


# --------------------------------------------------------------------------- #
# Reporting artifacts
# --------------------------------------------------------------------------- #
def _write_markdown(report, model, result, path: Path) -> None:
    strength = model.strength_table()
    strength = strength[strength["team"].isin(report.group)].reset_index(drop=True)
    lines = [
        f"# {report.club} — promotion forecast, {report.target_season}",
        "",
        f"**Generated:** {report.generated_at}",
        "",
        f"## Headline",
        "",
        f"> **{report.promotion_probability:.1%}** probability of promotion "
        f"(±{report.monte_carlo_se:.1%} Monte-Carlo SE, {result.n_sims:,} simulations).",
        "",
        "| Route | Probability |",
        "|---|---|",
        f"| Direct (champion) | {report.p_direct:.1%} |",
        f"| Reached play-off (2nd–5th) | {report.p_playoff_reached:.1%} |",
        f"| Won territorial play-off | {report.p_playoff_won:.1%} |",
        f"| **Promoted (any route)** | **{report.promotion_probability:.1%}** |",
        "",
        f"- Mean finishing position: **{report.mean_position:.1f}** of {len(report.group)}",
        f"- Mean points: **{report.mean_points:.1f}** "
        f"(p10 {result.points_pctiles['p10']:.0f} / p50 {result.points_pctiles['p50']:.0f} "
        f"/ p90 {result.points_pctiles['p90']:.0f})",
        f"- Pre-season strength rank within the group: "
        f"**{report.strength_rank_in_group}** of {len(report.group)}",
        "",
        "## Projected group strength (Dixon-Coles net rating)",
        "",
        "| # | Team | Attack | Defense | Net |",
        "|---|---|---|---|---|",
    ]
    for i, r in strength.iterrows():
        lines.append(
            f"| {i+1} | {r['team']} | {r['attack']:+.2f} | "
            f"{r['defense']:+.2f} | {r['net_strength']:+.2f} |"
        )
    if report.backtest:
        b = report.backtest
        lines += [
            "",
            "## Model validation (walk-forward)",
            "",
            f"- Matches scored out-of-sample: **{b['n_matches']:,}**",
            f"- Log-loss: **{b['log_loss']}** (baseline {b['baseline_log_loss']})",
            f"- Brier: **{b['brier']}** (baseline {b['baseline_brier']})",
            f"- RPS: **{b['rps']}**",
            f"- Top-pick accuracy: **{b['accuracy']:.1%}**",
        ]
    if report.squad:
        lines += [
            "",
            f"## Squad ({len(report.squad)} players)",
            "",
            "| Player | Position | Age | Nationality | Market value |",
            "|---|---|---|---|---|",
        ]
        for p in report.squad:
            mv = (
                f"€{p['market_value_eur']:,.0f}"
                if p.get("market_value_eur")
                else "—"
            )
            lines.append(
                f"| {p['name']} | {p.get('position') or ''} | "
                f"{p.get('age') or ''} | {p.get('nationality') or ''} | {mv} |"
            )
    if report.notes:
        lines += ["", "## Notes & caveats", ""]
        lines += [f"- {n}" for n in report.notes]
    lines += [
        "",
        "---",
        "*Model: time-weighted, L2-regularized Dixon-Coles goals model + "
        "Monte-Carlo season simulation. Data: Sofascore (results & standings). "
        "See README for methodology and limitations.*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_plots(report, model, result, group, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) finishing-position distribution
    pos = np.array(report.position_distribution)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#c62828" if i == 0 else ("#ef9a9a" if i < 5 else "#90a4ae")
              for i in range(len(pos))]
    ax.bar(np.arange(1, len(pos) + 1), pos * 100, color=colors)
    ax.set_xlabel("Final position")
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"{report.club} — simulated finishing position, {report.target_season}")
    ax.axvline(5.5, color="k", ls="--", lw=0.8)
    ax.text(5.6, ax.get_ylim()[1] * 0.9, "play-off cut", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "position_distribution.png", dpi=130)
    plt.close(fig)

    # 2) group net-strength
    strength = model.strength_table()
    strength = strength[strength["team"].isin(group)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    cols = ["#c62828" if t == report.club else "#546e7a" for t in strength["team"]]
    ax.barh(strength["team"][::-1], strength["net_strength"][::-1], color=cols[::-1])
    ax.set_xlabel("Dixon-Coles net strength (attack − defense)")
    ax.set_title(f"Projected {report.target_season} group strength")
    fig.tight_layout()
    fig.savefig(out_dir / "group_strength.png", dpi=130)
    plt.close(fig)

    # 3) calibration
    if report.backtest and report.backtest["calibration"]:
        cal = report.backtest["calibration"]
        xs = sorted(float(k) for k in cal)
        ys = [cal[k] if k in cal else cal[str(k)] for k in xs]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(xs, ys, "o-", color="#c62828", label="model")
        ax.set_xlabel("Predicted P(home win)")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration (walk-forward)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "calibration.png", dpi=130)
        plt.close(fig)
