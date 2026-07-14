"""Goalscorer data layer.

Sofascore's ``incidents`` endpoint exposes the scorer of every goal even at
tier 5 (line-ups and match stats do not exist there). This module scrapes goal
incidents for whole league-seasons and assembles a tidy per-goal table plus
per-player season tallies — the raw material for the goalscorer model.

We record for each goal: season, match, minute, scoring team, player name and
id, and whether it was a penalty or own goal. Own goals are excluded from a
player's *scoring* credit (they are tracked separately).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config
from .sofascore import SofascoreClient


def _extract_goals(incidents_json: dict, event: dict) -> list[dict]:
    """Pull goal rows out of one match's incidents payload."""
    if not incidents_json or "incidents" not in incidents_json:
        return []
    home = event["homeTeam"]["name"]
    away = event["awayTeam"]["name"]
    season = (event.get("season") or {}).get("year")
    eid = event.get("id")
    rows = []
    for inc in incidents_json["incidents"]:
        if inc.get("incidentType") != "goal":
            continue
        is_home = bool(inc.get("isHome"))
        scoring_team = home if is_home else away
        player = inc.get("player") or {}
        cls = inc.get("incidentClass")  # 'regular', 'penalty', 'ownGoal', ...
        is_own = cls == "ownGoal"
        rows.append(
            {
                "event_id": eid,
                "season_year": season,
                "minute": inc.get("time"),
                "team": scoring_team,
                "is_home": is_home,
                "player": player.get("name"),
                "player_id": player.get("id"),
                "penalty": cls == "penalty",
                "own_goal": is_own,
            }
        )
    return rows


def collect_goals(
    season_labels: list[str],
    client: SofascoreClient | None = None,
    team_filter: str | None = None,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scrape goal incidents for the given seasons.

    ``team_filter`` (substring) restricts to matches involving that team — use
    it to scrape only the target club's matches cheaply. Leave ``None`` to build
    the full league goalscorer dataset (heavier: one request per match).
    """
    client = client or SofascoreClient(verbose=False)
    season_map = client.list_seasons() or config.SEASON_IDS
    all_rows: list[dict] = []
    seen_events: set[int] = set()
    n_matches = 0
    for label in season_labels:
        sid = season_map.get(label) or config.SEASON_IDS.get(label)
        if sid is None:
            continue
        events = client.season_events(sid)
        # dedupe: an event can appear in both a league page and a filter
        events = [e for e in events if e["id"] not in seen_events]
        if team_filter:
            events = [
                e
                for e in events
                if team_filter.lower() in e["homeTeam"]["name"].lower()
                or team_filter.lower() in e["awayTeam"]["name"].lower()
            ]
        if verbose:
            print(f"[players] {label}: scraping incidents for {len(events)} matches",
                  flush=True)
        for k, e in enumerate(events, 1):
            seen_events.add(e["id"])
            inc = client.incidents(e["id"], force_refresh=force_refresh)
            rows = _extract_goals(inc, e)
            for r in rows:
                r["season"] = label
            all_rows.extend(rows)
            n_matches += 1
            if verbose and k % 50 == 0:
                print(f"[players]   {label}: {k}/{len(events)} matches, "
                      f"{len(all_rows)} goals so far", flush=True)
    if verbose:
        print(f"[players] done: {n_matches} matches, {len(all_rows)} goal rows",
              flush=True)
    return pd.DataFrame(all_rows)


LEAGUE_GOALS_PATH = config.PROCESSED_DIR / "goals_league_recent.json"


def load_league_goals(
    season_labels: list[str] | None = None,
    force_refresh: bool = False,
    prefer_cache: bool = True,
) -> pd.DataFrame:
    """Load league-wide goal incidents, from the committed cache if available.

    The scrape is expensive (one request per match), so we ship the recent
    seasons' goals under ``data/processed`` and read them by default.
    """
    if prefer_cache and not force_refresh and LEAGUE_GOALS_PATH.exists():
        return pd.read_json(LEAGUE_GOALS_PATH)
    labels = season_labels or ["24/25", "25/26"]
    df = collect_goals(labels, force_refresh=force_refresh)
    if not df.empty:
        df.to_json(LEAGUE_GOALS_PATH, orient="records")
    return df


def player_goal_tallies(goals: pd.DataFrame) -> pd.DataFrame:
    """Per-player, per-season goal counts (excluding own goals; named only)."""
    g = goals[(~goals["own_goal"]) & goals["player"].notna()].copy()
    tally = (
        g.groupby(["player", "player_id", "team", "season"])
        .agg(goals=("player", "size"), penalties=("penalty", "sum"))
        .reset_index()
    )
    return tally.sort_values(["season", "goals"], ascending=[True, False])


def team_goal_totals(goals: pd.DataFrame) -> pd.DataFrame:
    """Goals scored per team per season (from incidents; own goals credited)."""
    return (
        goals.groupby(["team", "season"])
        .size()
        .reset_index(name="goals_from_incidents")
    )


def team_penalty_info(goals: pd.DataFrame, team_substring: str = "Bin") -> dict:
    """Penalty fraction, designated taker, and own-goals for a team.

    Uses incidents (penalty flag is reliably set even when the scorer name is
    not). Penalty *fraction* is penalties / all team goals; the taker is the most
    frequent named penalty scorer.
    """
    g = goals[goals["team"].str.contains(team_substring, case=False, na=False)]
    if g.empty:
        return {"penalty_fraction": 0.10, "penalty_taker": None, "own_goals": 0}
    n_total = len(g)
    n_pen = int(g["penalty"].sum())
    pen_named = g[g["penalty"] & g["player"].notna()]
    taker = (
        pen_named["player"].value_counts().idxmax()
        if not pen_named.empty
        else None
    )
    # Sofascore under-flags penalties at this tier (names/flags missing on many
    # goals), so floor at a realistic league rate (~6-8% of goals are penalties).
    observed = n_pen / max(1, n_total)
    return {
        "penalty_fraction": round(max(observed, 0.06), 3),
        "penalty_taker": taker,
        "own_goals": int(g["own_goal"].sum()),
    }


@dataclass
class PlayerScoring:
    """A player's shrunk scoring rate as a share of their team's goals."""

    player: str
    team: str
    total_goals: int
    seasons_played: int
    share_of_team_goals: float   # shrunk empirical share
    weight: float                # unnormalized scoring propensity
