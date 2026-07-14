"""Assemble Sofascore JSON into tidy pandas structures.

Produces two things the model needs:

* ``matches`` — one row per finished league match (date, season, round, home,
  away, goals). This is the training data for the ratings models.
* ``standings`` — the final/current league table for a given season.

A bundled snapshot (``data/processed/snapshot_25_26.json``) lets the whole
pipeline run offline if Sofascore is unreachable, so results stay reproducible.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .sofascore import SofascoreClient

SNAPSHOT_PATH = config.PROCESSED_DIR / "snapshot_25_26.json"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _event_to_row(ev: dict) -> dict | None:
    """Convert a Sofascore event dict into a flat match row, or None if unusable."""
    status = ev.get("status", {})
    if status.get("code") != config.STATUS_FINISHED:
        return None
    home_score = (ev.get("homeScore") or {}).get("current")
    away_score = (ev.get("awayScore") or {}).get("current")
    if home_score is None or away_score is None:
        return None
    ts = ev.get("startTimestamp")
    date = (
        datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        if ts
        else None
    )
    return {
        "event_id": ev.get("id"),
        "date": date,
        "timestamp": ts,
        "round": (ev.get("roundInfo") or {}).get("round"),
        "season_year": (ev.get("season") or {}).get("year"),
        "home_id": ev["homeTeam"]["id"],
        "home": ev["homeTeam"]["name"],
        "away_id": ev["awayTeam"]["id"],
        "away": ev["awayTeam"]["name"],
        "home_goals": int(home_score),
        "away_goals": int(away_score),
    }


def matches_from_events(events: list[dict]) -> pd.DataFrame:
    rows = [r for ev in events if (r := _event_to_row(ev)) is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="event_id").sort_values("timestamp")
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def standings_from_json(data: dict) -> pd.DataFrame:
    rows = data["standings"][0]["rows"]
    recs = []
    for r in rows:
        recs.append(
            {
                "position": r["position"],
                "team_id": r["team"]["id"],
                "team": r["team"]["name"],
                "played": r["matches"],
                "wins": r["wins"],
                "draws": r["draws"],
                "losses": r["losses"],
                "goals_for": r["scoresFor"],
                "goals_against": r["scoresAgainst"],
                "goal_diff": r["scoresFor"] - r["scoresAgainst"],
                "points": r["points"],
            }
        )
    return pd.DataFrame(recs).sort_values("position").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# High-level collection
# --------------------------------------------------------------------------- #
def collect(
    seasons: list[str] | None = None,
    client: SofascoreClient | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fetch matches + standings for the requested seasons.

    Returns ``(all_matches, latest_standings, standings_by_season)``.
    ``seasons`` are labels from :data:`config.SEASON_IDS` (default: all).
    """
    client = client or SofascoreClient()
    season_map = client.list_seasons() or config.SEASON_IDS
    if seasons is None:
        seasons = list(config.SEASON_IDS.keys())

    frames: list[pd.DataFrame] = []
    standings_by_season: dict[str, pd.DataFrame] = {}
    for label in seasons:
        sid = season_map.get(label) or config.SEASON_IDS.get(label)
        if sid is None:
            continue
        events = client.season_events(sid, force_refresh=force_refresh)
        m = matches_from_events(events)
        if not m.empty:
            m = m.copy()
            m["season"] = label
            frames.append(m)
        st_json = client.standings(sid, force_refresh=force_refresh)
        if st_json:
            standings_by_season[label] = standings_from_json(st_json)

    all_matches = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    if not all_matches.empty:
        all_matches = all_matches.sort_values("timestamp").reset_index(drop=True)

    # latest season with a standings table = the most recent completed season
    latest_label = next(
        (s for s in config.SEASON_IDS if s in standings_by_season), None
    )
    latest_standings = (
        standings_by_season.get(latest_label, pd.DataFrame())
    )
    return all_matches, latest_standings, standings_by_season


# --------------------------------------------------------------------------- #
# Snapshot (offline reproducibility)
# --------------------------------------------------------------------------- #
def save_snapshot(
    matches: pd.DataFrame,
    latest_standings: pd.DataFrame,
    path: Path = SNAPSHOT_PATH,
) -> None:
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "matches": json.loads(matches.to_json(orient="records", date_format="iso")),
        "latest_standings": json.loads(
            latest_standings.to_json(orient="records")
        ),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_snapshot(
    path: Path = SNAPSHOT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    matches = pd.DataFrame(payload["matches"])
    if not matches.empty:
        matches["date"] = pd.to_datetime(matches["date"])
    standings = pd.DataFrame(payload["latest_standings"])
    return matches, standings


def project_target_group(
    latest_standings: pd.DataFrame,
    league_size: int = config.LEAGUE_SIZE,
    n_relegated: int = 3,
    newcomer_prefix: str = "Newcomer",
) -> list[str]:
    """Best-effort composition of the *target* (upcoming) season's group.

    The official group for 26/27 is not published while the transfer window is
    open, so we reconstruct it: keep the teams that neither won direct promotion
    (1st) nor were relegated (bottom ``n_relegated``) from the latest completed
    season, then top up to ``league_size`` with generic newcomer placeholders
    (promoted from Regional Preferente / dropped from Segunda Fed.). Newcomers
    are rated as league-average-with-penalty by the simulator.
    """
    st = latest_standings.sort_values("position")
    champion = st.iloc[0]["team"]
    relegated = set(st.iloc[-n_relegated:]["team"])
    returning = [
        t for t in st["team"]
        if t != champion and t not in relegated
    ]
    newcomers = [
        f"{newcomer_prefix} {i + 1}"
        for i in range(max(0, league_size - len(returning)))
    ]
    return returning + newcomers


def load(
    prefer_snapshot: bool = False, force_refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience loader with graceful offline fallback.

    Tries the live API; on failure (or when ``prefer_snapshot``) falls back to
    the bundled snapshot.
    """
    if prefer_snapshot and SNAPSHOT_PATH.exists():
        return load_snapshot()
    try:
        matches, latest, _ = collect(force_refresh=force_refresh)
        if matches.empty:
            raise RuntimeError("no matches returned")
        return matches, latest
    except Exception as exc:  # network blocked etc.
        if SNAPSHOT_PATH.exists():
            print(f"[data] live fetch failed ({exc}); using snapshot")
            return load_snapshot()
        raise
