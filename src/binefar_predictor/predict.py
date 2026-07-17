"""End-to-end orchestration: data -> ratings -> simulation -> goalscorers -> report.

``run_prediction`` is the single entry point used by the CLI. It runs the whole
pipeline and writes ``models/prediction.json``, ``models/report.md`` and a set of
diagnostic plots. Every enrichment (squad, goalscorers, pichichi, sensitivity,
backtest) degrades gracefully if its data source is unavailable — the core
promotion probability always computes from the bundled snapshot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data
from .ratings import DixonColesModel, EloModel, PiRatingModel


@dataclass
class PredictionReport:
    club: str
    target_season: str
    generated_at: str
    n_matches_trained: int
    n_seasons: int
    # promotion
    promotion_probability: float          # ensemble mean
    promotion_range: list[float]          # [min, max] across ensemble members
    ensemble_members: dict
    ensemble_weights: dict
    p_direct: float
    p_playoff_reached: float
    p_playoff_won: float
    monte_carlo_se: float
    mean_position: float
    mean_points: float
    mean_goals_for: float
    position_distribution: list[float]
    strength_rank_in_group: int
    strength_agreement: dict
    group: list[str]
    newcomers: list[str]
    projected_table: list
    model_params: dict
    squad: list[dict] = field(default_factory=list)
    goalscorers: list[dict] = field(default_factory=list)
    anytime_scorers: list[dict] = field(default_factory=list)
    pichichi_race: list[dict] = field(default_factory=list)
    goleadores_crosscheck: list[dict] = field(default_factory=list)
    sensitivity: list[dict] = field(default_factory=list)
    form: dict = field(default_factory=dict)
    backtest: dict | None = None
    bootstrap: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)

    def headline(self) -> str:
        lo, hi = self.promotion_range
        return (f"{self.club} — probability of promotion in {self.target_season}: "
                f"{self.promotion_probability:.1%} (ensemble range {lo:.1%}–{hi:.1%})")


def _group_strength_rank(model: DixonColesModel, group: list[str], team: str) -> int:
    net = lambda t: model.attack.get(t, 0.0) - model.defense.get(t, 0.0)
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
    include_goalscorers: bool = True,
    include_sensitivity: bool = True,
    include_backtest: bool = True,
    include_bootstrap: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> PredictionReport:
    from .simulate import SeasonSimulator
    from .ensemble import run_ensemble, bootstrap_promotion

    def _say(*a):
        if verbose:
            print(*a, flush=True)

    notes: list[str] = []

    _say("[1/8] Loading match data …")
    matches, latest = data.load(prefer_snapshot=prefer_snapshot, force_refresh=force_refresh)
    n_seasons = matches["season"].nunique() if "season" in matches else 0
    _say(f"      {len(matches):,} matches, {n_seasons} seasons.")

    _say("[2/8] Identifying real promoted teams (new joiners) …")
    promoted = [] if prefer_snapshot else data.fetch_promoted_teams()
    group = data.project_target_group(latest, promoted_teams=promoted)
    newcomers = [t for t in group if t not in set(latest["team"])]
    if promoted:
        _say(f"      newcomers from Regional Preferente: {promoted}")
    else:
        notes.append("Could not fetch real promoted teams; used placeholders.")
    if target not in group:
        group = group[:-1] + [target]
        notes.append(f"{target} forced into group (was absent).")

    _say("[3/8] Fitting ratings + running model ensemble …")
    model = DixonColesModel(half_life_days=half_life_days, l2=l2).fit(matches)
    elo = EloModel().fit(matches)
    pi = PiRatingModel().fit(matches)
    ens = run_ensemble(matches, group, target=target, n_sims=min(n_sims, 25_000))
    _say("      " + ens.summary().replace("\n", "\n      "))

    _say(f"[4/8] Monte-Carlo simulating {n_sims:,} seasons (primary model) …")
    sim = SeasonSimulator(model, group, target=target)
    result = sim.run(n_sims=n_sims)
    _say("      " + result.summary().replace("\n", "\n      "))

    bootstrap_dict = None
    if include_bootstrap:
        _say("      bootstrapping parameter uncertainty …")
        try:
            bootstrap_dict = bootstrap_promotion(matches, group, target=target)
            _say(f"      parameter-uncertainty 90% CI: "
                 f"{bootstrap_dict['ci90'][0]:.1%}–{bootstrap_dict['ci90'][1]:.1%} "
                 f"(mean {bootstrap_dict['mean']:.1%})")
        except Exception as exc:
            notes.append(f"Bootstrap skipped: {exc}")

    # ---- squad + goalscorers ------------------------------------------- #
    squad_records, goalscorers, anytime, pichichi, crosscheck = [], [], [], [], []
    if include_squad or include_goalscorers:
        _say("[5/8] Squad + goalscorer model …")
        try:
            from . import transfermarkt as tm, futbolme, players as P
            from .goalscorer import GoalscorerModel, league_top_scorer, match_goals_to_squad

            players_tm = tm.fetch_squad(season_year=int("20" + config.TARGET_SEASON[:2]))
            if not players_tm:
                players_tm = tm.fetch_squad(season_year=2025)
            squad_df = tm.squad_dataframe(players_tm)
            squad_records = [asdict(p) for p in players_tm]
            _say(f"      squad: {len(players_tm)} players")

            goleadores = pd.DataFrame() if prefer_snapshot else futbolme.fetch_top_scorers()
            league_goals = P.load_league_goals()
            if include_goalscorers and not squad_df.empty:
                if goleadores.empty:
                    # offline fallback: build tallies from cached incidents
                    tallies = P.player_goal_tallies(league_goals)
                    tallies = tallies[tallies.season == "25/26"]
                    goleadores = tallies.rename(columns={})[["player", "team", "goals"]]
                    notes.append("Goleadores from cached incidents (offline); "
                                 "counts undercount vs Futbolme.")
                team_gol = goleadores[goleadores["team"].str.contains("Bin", case=False, na=False)]
                pg = match_goals_to_squad(list(squad_df["name"]), team_gol, goleadores)
                peninfo = P.team_penalty_info(league_goals, "Bin")
                gm = GoalscorerModel(
                    squad=squad_df, player_goals=pg,
                    penalty_taker=peninfo["penalty_taker"],
                    penalty_fraction=peninfo["penalty_fraction"],
                    own_goals=peninfo["own_goals"],
                ).fit()
                alloc = gm.allocate(result.target_goals_for)
                goalscorers = alloc.to_dict("records")
                mean_gpm = float(result.target_goals_for.mean())
                anytime = gm.anytime_scorer_table(mean_gpm / config.MATCHES_PER_TEAM).head(12).to_dict("records")
                _say(f"      top predicted scorer: {alloc.iloc[0]['player']} "
                     f"(~{alloc.iloc[0]['exp_goals']:.0f} goals, "
                     f"{alloc.iloc[0]['p_team_top_scorer']:.0%} to lead the team)")
                # league pichichi across all teams
                if not goleadores.empty and result.all_goals_for is not None:
                    pichichi = league_top_scorer(result.all_goals_for, group, goleadores).to_dict("records")
                # cross-check our matched Binéfar scorers vs Futbolme table
                if not team_gol.empty:
                    crosscheck = team_gol.sort_values("goals", ascending=False).head(8).to_dict("records")
        except Exception as exc:
            notes.append(f"Goalscorer step failed: {exc}")
            _say(f"      (goalscorer step skipped: {exc})")

    # ---- sensitivity + form ------------------------------------------- #
    sensitivity_records, form = [], {}
    if include_sensitivity:
        _say("[6/8] Sensitivity analysis + form tables …")
        try:
            from .analysis import sensitivity as _sens, form_tables
            sensitivity_records = _sens(matches, latest, target=target, n_sims=12_000).to_dict("records")
            form = form_tables(matches, target)
        except Exception as exc:
            notes.append(f"Sensitivity/form failed: {exc}")

    # ---- backtest ------------------------------------------------------ #
    backtest_dict = None
    if include_backtest:
        _say("[7/8] Walk-forward validation …")
        from .evaluate import match_backtest
        rep = match_backtest(matches, half_life_days=half_life_days, l2=l2, refit_every=60)
        backtest_dict = {
            "n_matches": rep.n_matches, "log_loss": round(rep.log_loss, 4),
            "brier": round(rep.brier, 4), "rps": round(rep.rps, 4),
            "baseline_log_loss": round(rep.baseline_log_loss, 4),
            "baseline_brier": round(rep.baseline_brier, 4),
            "accuracy": round(rep.accuracy, 4), "calibration": rep.calibration,
        }
        _say("      " + rep.summary().replace("\n", "\n      "))
        # champion backtest: did the model's pre-season favourite match reality?
        try:
            from .evaluate import champion_backtest
            sbs = data.load_standings_by_season()
            if sbs:
                cb = champion_backtest(matches, sbs, half_life_days=half_life_days,
                                       l2=l2, n_sims=4000)
                if not cb.empty:
                    backtest_dict["champion"] = {
                        "seasons": int(len(cb)),
                        "mean_p_direct_to_champion": round(float(cb["model_p_direct"].mean()), 3),
                        "mean_predicted_pos_of_champion": round(float(cb["model_mean_pos"].mean()), 2),
                        "base_rate": round(1.0 / config.LEAGUE_SIZE, 3),
                    }
                    _say(f"      champion backtest: model gave eventual champions "
                         f"{backtest_dict['champion']['mean_p_direct_to_champion']:.0%} avg "
                         f"pre-season title prob (base rate ~6%), "
                         f"predicted pos ~{backtest_dict['champion']['mean_predicted_pos_of_champion']}")
        except Exception as exc:
            notes.append(f"Champion backtest skipped: {exc}")

    mean_goals_for = float(result.target_goals_for.mean()) if result.target_goals_for is not None else 0.0

    _say("[8/8] Writing report + plots …")
    report = PredictionReport(
        club=target, target_season=config.TARGET_SEASON,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        n_matches_trained=len(matches), n_seasons=int(n_seasons),
        promotion_probability=ens.mean, promotion_range=[ens.min, ens.max],
        ensemble_members=ens.members,
        ensemble_weights=ens.weights,
        p_direct=result.p_direct, p_playoff_reached=result.p_playoff_reached,
        p_playoff_won=result.p_playoff_won, monte_carlo_se=result.se,
        mean_position=result.mean_position, mean_points=result.mean_points,
        mean_goals_for=mean_goals_for,
        position_distribution=result.position_dist.tolist(),
        strength_rank_in_group=_group_strength_rank(model, group, target),
        strength_agreement=ens.strength_agreement,
        group=group, newcomers=newcomers,
        projected_table=result.league_table or [],
        model_params={
            "half_life_days": half_life_days, "l2": l2,
            "home_advantage": round(model.home, 4), "mu": round(model.mu, 4),
            "rho": round(model.rho, 4),
            "national_phase_conversion": config.NATIONAL_PHASE_CONVERSION,
            "newcomer_net_strength_penalty": config.NEWCOMER_NET_STRENGTH_PENALTY,
            "elo_binefar": round(elo.ratings.get(target, elo.base_rating), 1),
            "pi_binefar": round(pi.strength(target), 3),
        },
        squad=squad_records, goalscorers=goalscorers, anytime_scorers=anytime,
        pichichi_race=pichichi, goleadores_crosscheck=crosscheck,
        sensitivity=sensitivity_records, form=form,
        backtest=backtest_dict, bootstrap=bootstrap_dict, notes=notes,
    )
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_json(config.MODELS_DIR / "prediction.json")
    _write_markdown(report, model, result, config.MODELS_DIR / "report.md")
    if make_plots:
        _make_plots(report, model, result, group, config.MODELS_DIR)
    build_dashboard()
    _say(f"\nSaved: {config.MODELS_DIR}/prediction.json, report.md"
         + (", plots" if make_plots else "") + ", web/index.html")
    return report


def build_dashboard() -> bool:
    """(Re)build the interactive web dashboard from the latest prediction.json."""
    try:
        import importlib.util

        path = config.PROJECT_ROOT / "web" / "build_dashboard.py"
        spec = importlib.util.spec_from_file_location("build_dashboard", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build()
        return True
    except Exception as exc:
        print(f"[dashboard] skipped: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _write_markdown(report, model, result, path: Path) -> None:
    strength = model.strength_table()
    strength = strength[strength["team"].isin(report.group)].reset_index(drop=True)
    lo, hi = report.promotion_range
    L = [
        f"# {report.club} — promotion forecast, {report.target_season}",
        "", f"**Generated:** {report.generated_at}", "",
        "## Headline", "",
        f"> **{report.promotion_probability:.1%}** probability of promotion "
        f"(ensemble range **{lo:.1%}–{hi:.1%}**; primary-model Monte-Carlo SE "
        f"±{report.monte_carlo_se:.1%}, {result.n_sims:,} sims).", "",
    ]
    if report.bootstrap:
        b = report.bootstrap
        L += [f"> Accounting for **rating-estimation uncertainty** (bootstrap, {b['n_boot']} "
              f"resamples), the parameter-uncertainty 90% interval is **{b['ci90'][0]:.1%}–{b['ci90'][1]:.1%}** "
              f"(mean {b['mean']:.1%}) — much wider, because tier-5 ratings come from small "
              f"samples. This band covers estimation error only; the play-off conversion rate, "
              f"newcomer priors and the unpublished 26/27 group add further uncertainty on top "
              f"(see Sensitivity). Read the headline as a central estimate, not a precise number.", ""]
    L += [
        "| Route | Probability |", "|---|---|",
        f"| Direct (champion) | {report.p_direct:.1%} |",
        f"| Reached play-off (2nd–5th) | {report.p_playoff_reached:.1%} |",
        f"| Won territorial play-off | {report.p_playoff_won:.1%} |",
        f"| **Promoted (any route)** | **{report.promotion_probability:.1%}** |", "",
        f"- Mean finishing position: **{report.mean_position:.1f}** of {len(report.group)}",
        f"- Mean points: **{report.mean_points:.1f}**; mean goals scored: "
        f"**{report.mean_goals_for:.1f}**",
        f"- Pre-season strength rank in group: **{report.strength_rank_in_group}** "
        f"of {len(report.group)}", "",
        "## Ensemble members", "",
        "Members are weighted by out-of-sample (last-season) W/D/L log-loss, so "
        "the better-calibrated models count more toward the headline number.", "",
        "| Model variant | Promotion prob | Weight |", "|---|---|---|",
    ]
    for name, p in report.ensemble_members.items():
        w = report.ensemble_weights.get(name, 0)
        L.append(f"| {name} | {p:.1%} | {w:.0%} |")
    if report.strength_agreement:
        L += ["", "Cross-model strength-ordering agreement (Spearman): "
              + ", ".join(f"{k} {v:+.2f}" for k, v in report.strength_agreement.items())]

    L += ["", f"## New joiners in the group ({report.target_season})", "",
          "Teams promoted up from Regional Preferente Aragón (real names where "
          "resolvable; UD Fraga & CD Brea carry real Tercera ratings from history, "
          "the rest use the newcomer prior):", "",
          "- " + ", ".join(report.newcomers) if report.newcomers else "- (none resolved)"]

    L += ["", "## Projected group strength (Dixon-Coles net rating)", "",
          "| # | Team | Attack | Defense | Net |", "|---|---|---|---|---|"]
    for i, r in strength.iterrows():
        L.append(f"| {i+1} | {r['team']} | {r['attack']:+.2f} | "
                 f"{r['defense']:+.2f} | {r['net_strength']:+.2f} |")

    if report.projected_table:
        L += ["", "## Projected final table (mean over simulations)", "",
              "| # | Team | Mean pts | Mean pos | P(champion) | P(top-5) |",
              "|---|---|---|---|---|---|"]
        for i, r in enumerate(report.projected_table, 1):
            mark = " **←**" if r.get("is_target") else ""
            L.append(f"| {i} | {r['team']}{mark} | {r['mean_points']} | "
                     f"{r['mean_position']} | {r['p_champion']:.0%} | {r['p_top5']:.0%} |")

    if report.goalscorers:
        L += ["", "## Predicted goalscorers — CD Binéfar", "",
              "| Player | Pos | Goals 25/26 | Exp goals 26/27 | P10–P90 | P(team top scorer) |",
              "|---|---|---|---|---|---|"]
        for g in report.goalscorers[:12]:
            L.append(f"| {g['player']} | {g.get('position') or ''} | "
                     f"{g['goals_last_season']} | {g['exp_goals']} | "
                     f"{g['p10']}–{g['p90']} | {g['p_team_top_scorer']:.0%} |")

    if report.pichichi_race:
        L += ["", "## Pichichi race (top scorer of the whole group)", "",
              "| Player (team) | Exp goals | P(pichichi) |", "|---|---|---|"]
        for g in report.pichichi_race[:12]:
            L.append(f"| {g['player_team']} | {g['exp_goals']} | {g['p_pichichi']:.0%} |")

    if report.anytime_scorers:
        L += ["", "## Anytime-scorer probability (typical match)", "",
              "| Player | P(scores) |", "|---|---|"]
        for a in report.anytime_scorers[:8]:
            L.append(f"| {a['player']} | {a['p_anytime']:.0%} |")

    if report.sensitivity:
        L += ["", "## Sensitivity of the promotion probability", "",
              "| Parameter | Value | Promotion prob |", "|---|---|---|"]
        for s in report.sensitivity:
            L.append(f"| {s['parameter']} | {s['value']} | {s['p_promotion']:.1%} |")

    if report.form:
        bv = report.form.get("by_venue", {})
        L += ["", "## Recent form & splits", "",
              f"- Last 10: {report.form.get('last10_record')} "
              f"(PPG {report.form.get('last10_ppg')})"]
        for venue, d in bv.items():
            L.append(f"- {venue.title()}: {d.get('ppg')} PPG, "
                     f"{d.get('gf')} GF / {d.get('ga')} GA per game")

    if report.backtest:
        b = report.backtest
        L += ["", "## Model validation (walk-forward)", "",
              f"- Matches scored out-of-sample: **{b['n_matches']:,}**",
              f"- Log-loss **{b['log_loss']}** (baseline {b['baseline_log_loss']})",
              f"- Brier **{b['brier']}** (baseline {b['baseline_brier']})",
              f"- RPS **{b['rps']}**; top-pick accuracy **{b['accuracy']:.1%}**"]
        if b.get("champion"):
            c = b["champion"]
            L.append(f"- Champion backtest ({c['seasons']} seasons): the model gave the "
                     f"eventual champion **{c['mean_p_direct_to_champion']:.0%}** average "
                     f"pre-season title probability (base rate {c['base_rate']:.0%}) and "
                     f"predicted them ~**{c['mean_predicted_pos_of_champion']:.1f}th** on average.")

    if report.squad:
        L += ["", f"## Squad ({len(report.squad)} players)", "",
              "| Player | Position | Age | Nationality |", "|---|---|---|---|"]
        for p in report.squad:
            L.append(f"| {p['name']} | {p.get('position') or ''} | "
                     f"{p.get('age') or ''} | {p.get('nationality') or ''} |")

    if report.notes:
        L += ["", "## Notes & caveats", ""] + [f"- {n}" for n in report.notes]
    L += ["", "---",
          "*Ensemble of time-weighted Dixon-Coles / independent-Poisson goals "
          "models + Monte-Carlo season simulation + Dirichlet-multinomial "
          "goalscorer allocation. Data: Sofascore (results, standings, goal "
          "incidents), Futbolme (goleadores), Transfermarkt (squad), Regional "
          "Preferente (promoted teams). See README.*"]
    path.write_text("\n".join(L), encoding="utf-8")


def _make_plots(report, model, result, group, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.array(report.position_distribution)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#c62828" if i == 0 else ("#ef9a9a" if i < 5 else "#90a4ae") for i in range(len(pos))]
    ax.bar(np.arange(1, len(pos) + 1), pos * 100, color=colors)
    ax.set_xlabel("Final position"); ax.set_ylabel("Probability (%)")
    ax.set_title(f"{report.club} — simulated finishing position, {report.target_season}")
    ax.axvline(5.5, color="k", ls="--", lw=0.8)
    ax.text(5.6, ax.get_ylim()[1] * 0.9, "play-off cut", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "position_distribution.png", dpi=130); plt.close(fig)

    strength = model.strength_table()
    strength = strength[strength["team"].isin(group)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    cols = ["#c62828" if t == report.club else "#546e7a" for t in strength["team"]]
    ax.barh(strength["team"][::-1], strength["net_strength"][::-1], color=cols[::-1])
    ax.set_xlabel("Dixon-Coles net strength (attack − defense)")
    ax.set_title(f"Projected {report.target_season} group strength")
    fig.tight_layout(); fig.savefig(out_dir / "group_strength.png", dpi=130); plt.close(fig)

    if report.backtest and report.backtest["calibration"]:
        cal = report.backtest["calibration"]
        xs = sorted(float(k) for k in cal)
        ys = [cal[k] if k in cal else cal[str(k)] for k in xs]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(xs, ys, "o-", color="#c62828", label="model")
        ax.set_xlabel("Predicted P(home win)"); ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration (walk-forward)"); ax.legend()
        fig.tight_layout(); fig.savefig(out_dir / "calibration.png", dpi=130); plt.close(fig)

    if report.goalscorers:
        g = [x for x in report.goalscorers if not x["player"].startswith("(other")][:10]
        fig, ax = plt.subplots(figsize=(9, 5))
        names = [x["player"] for x in g][::-1]
        exp = [x["exp_goals"] for x in g][::-1]
        ax.barh(names, exp, color="#00695c")
        ax.set_xlabel("Expected goals, 26/27")
        ax.set_title(f"{report.club} — predicted goalscorers")
        fig.tight_layout(); fig.savefig(out_dir / "goalscorers.png", dpi=130); plt.close(fig)

    if report.sensitivity:
        df = pd.DataFrame(report.sensitivity)
        params = df["parameter"].unique()
        fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3.4), squeeze=False)
        for ax, p in zip(axes[0], params):
            sub = df[df["parameter"] == p]
            ax.plot(sub["value"].astype(str), sub["p_promotion"] * 100, "o-", color="#c62828")
            ax.set_title(p, fontsize=9); ax.set_ylabel("promo %")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        fig.suptitle("Promotion probability sensitivity", fontsize=11)
        fig.tight_layout(); fig.savefig(out_dir / "sensitivity.png", dpi=130); plt.close(fig)
