"""Tests for data assembly, group projection, and snapshot round-trip."""
import pandas as pd

from binefar_predictor import config, data


def _latest_standings():
    # 18 teams, position 1..18 (champion first, bottom three relegated)
    return pd.DataFrame(
        {
            "position": list(range(1, 19)),
            "team": [f"Team {i}" for i in range(1, 19)],
            "points": list(range(90, 90 - 18, -1)),
            "goals_for": [50] * 18,
            "goals_against": [30] * 18,
            "goal_diff": [20] * 18,
        }
    )


def test_project_group_drops_champion_and_relegated():
    st = _latest_standings()
    group = data.project_target_group(st, league_size=18, n_relegated=3,
                                      promoted_teams=["Up A", "Up B", "Up C", "Up D"])
    assert len(group) == 18
    assert "Team 1" not in group          # champion promoted out
    assert "Team 18" not in group         # relegated
    assert "Team 16" not in group and "Team 17" not in group  # relegated
    assert "Team 2" in group              # runner-up stays
    # real promoted teams fill the four freed slots
    for up in ["Up A", "Up B", "Up C", "Up D"]:
        assert up in group


def test_project_group_pads_with_placeholders_when_no_promoted():
    st = _latest_standings()
    group = data.project_target_group(st, promoted_teams=[])
    assert len(group) == 18
    assert any(t.startswith("Newcomer") for t in group)


def test_snapshot_roundtrip(tmp_path):
    matches = pd.DataFrame([
        {"event_id": 1, "date": "2025-09-01", "timestamp": 1.7e9, "round": 1,
         "season": "25/26", "home": "A", "away": "B", "home_goals": 2, "away_goals": 1},
    ])
    latest = _latest_standings()
    by_season = {"25/26": latest}
    path = tmp_path / "snap.json"
    data.save_snapshot(matches, latest, path=path, standings_by_season=by_season)
    m2, st2 = data.load_snapshot(path=path)
    assert len(m2) == 1 and m2.iloc[0]["home"] == "A"
    assert len(st2) == 18
    sbs = data.load_standings_by_season(path=path)
    assert "25/26" in sbs and len(sbs["25/26"]) == 18
