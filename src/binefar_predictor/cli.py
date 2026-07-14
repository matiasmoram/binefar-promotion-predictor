"""Command-line interface.

    python -m binefar_predictor predict            # full pipeline + report
    python -m binefar_predictor predict --offline  # use bundled snapshot
    python -m binefar_predictor scrape             # refresh cached data + snapshot
    python -m binefar_predictor backtest           # walk-forward validation only
    python -m binefar_predictor squad              # print current squad
"""
from __future__ import annotations

import argparse

from . import config


def _cmd_predict(args) -> None:
    from .predict import run_prediction

    report = run_prediction(
        target=args.team,
        n_sims=args.sims,
        half_life_days=args.half_life,
        l2=args.l2,
        prefer_snapshot=args.offline,
        force_refresh=args.refresh,
        include_squad=not args.no_squad,
        include_goalscorers=not args.no_goalscorers,
        include_sensitivity=not args.no_sensitivity,
        include_backtest=not args.no_backtest,
        make_plots=not args.no_plots,
    )
    print("\n" + "=" * 70)
    print(report.headline())
    print("=" * 70)


def _cmd_scrape(args) -> None:
    from . import data, players as P

    matches, latest, _ = data.collect(force_refresh=args.refresh)
    data.save_snapshot(matches, latest)
    print(f"Cached {len(matches):,} matches; snapshot written to "
          f"{data.SNAPSHOT_PATH}")
    if args.goals:
        print("Scraping league goal incidents (this is slow) …")
        goals = P.load_league_goals(force_refresh=args.refresh, prefer_cache=False)
        print(f"Cached {len(goals):,} goal rows to {P.LEAGUE_GOALS_PATH}")


def _cmd_backtest(args) -> None:
    from . import data
    from .evaluate import match_backtest

    matches, _ = data.load(prefer_snapshot=args.offline)
    rep = match_backtest(matches, half_life_days=args.half_life, l2=args.l2)
    print(rep.summary())


def _cmd_dashboard(args) -> None:
    from .predict import build_dashboard

    if build_dashboard():
        from . import config
        print(f"Built {config.PROJECT_ROOT/'web'/'index.html'} — open it in a browser.")


def _cmd_squad(args) -> None:
    from . import transfermarkt as tm

    players = tm.fetch_squad(season_year=args.season)
    if not players:
        print("No squad data available (Transfermarkt unreachable or empty).")
        return
    df = tm.squad_dataframe(players)
    print(df.to_string(index=False))
    print("\n", tm.squad_value_summary(players))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="binefar_predictor",
        description="Will CD Binéfar be promoted next season? A Monte-Carlo forecast.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("predict", help="run the full prediction pipeline")
    pp.add_argument("--team", default=config.CLUB_NAME)
    pp.add_argument("--sims", type=int, default=50_000)
    pp.add_argument("--half-life", type=float, default=365.0, dest="half_life")
    pp.add_argument("--l2", type=float, default=0.05)
    pp.add_argument("--offline", action="store_true", help="use bundled snapshot")
    pp.add_argument("--refresh", action="store_true", help="force re-download")
    pp.add_argument("--no-squad", action="store_true")
    pp.add_argument("--no-goalscorers", action="store_true")
    pp.add_argument("--no-sensitivity", action="store_true")
    pp.add_argument("--no-backtest", action="store_true")
    pp.add_argument("--no-plots", action="store_true")
    pp.set_defaults(func=_cmd_predict)

    ps = sub.add_parser("scrape", help="refresh cached data and snapshot")
    ps.add_argument("--refresh", action="store_true")
    ps.add_argument("--goals", action="store_true", help="also scrape goal incidents")
    ps.set_defaults(func=_cmd_scrape)

    pb = sub.add_parser("backtest", help="walk-forward validation only")
    pb.add_argument("--offline", action="store_true")
    pb.add_argument("--half-life", type=float, default=365.0, dest="half_life")
    pb.add_argument("--l2", type=float, default=0.05)
    pb.set_defaults(func=_cmd_backtest)

    pdash = sub.add_parser("dashboard", help="rebuild web/index.html from prediction.json")
    pdash.set_defaults(func=_cmd_dashboard)

    pq = sub.add_parser("squad", help="print the squad from Transfermarkt")
    pq.add_argument("--season", type=int, default=2025)
    pq.set_defaults(func=_cmd_squad)
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
