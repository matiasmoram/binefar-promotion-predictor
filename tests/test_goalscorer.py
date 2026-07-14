"""Tests for the goalscorer model."""
import numpy as np
import pandas as pd

from binefar_predictor.goalscorer import (
    GoalscorerModel,
    league_top_scorer,
    match_goals_to_squad,
)


def _squad():
    return pd.DataFrame([
        {"name": "Adrià de Mesa", "position": "Centre-Forward"},
        {"name": "Chicho Barreda", "position": "Second Striker"},
        {"name": "Álex Rico", "position": "Central Midfield"},
        {"name": "Andreu Lladonosa", "position": "Goalkeeper"},
    ])


def test_name_matching_within_team():
    gol = pd.DataFrame([
        {"player": "De Mesa", "team": "CD Binéfar", "goals": 14},
        {"player": "Chicho", "team": "CD Binéfar", "goals": 14},
        {"player": "Rico", "team": "CD Binéfar", "goals": 4},
    ])
    pg = match_goals_to_squad(list(_squad()["name"]), gol, gol)
    assert pg["Adrià de Mesa"] == 14
    assert pg["Chicho Barreda"] == 14
    assert pg["Álex Rico"] == 4
    assert pg["Andreu Lladonosa"] == 0


def test_no_false_firstname_match():
    # a league "Álex" from another team must NOT attach to "Álex Rico"
    team_gol = pd.DataFrame([{"player": "Rico", "team": "CD Binéfar", "goals": 4}])
    league = pd.DataFrame([
        {"player": "Rico", "team": "CD Binéfar", "goals": 4},
        {"player": "Álex", "team": "UD Casetas", "goals": 8},
    ])
    pg = match_goals_to_squad(list(_squad()["name"]), team_gol, league)
    assert pg["Álex Rico"] == 4  # not 8


def test_allocation_conserves_goals_and_ranks_striker_first():
    squad = _squad()
    pg = {"Adrià de Mesa": 14, "Chicho Barreda": 14, "Álex Rico": 4,
          "Andreu Lladonosa": 0}
    gm = GoalscorerModel(squad=squad, player_goals=pg, penalty_taker=None,
                         penalty_fraction=0.0).fit()
    team_goals = np.full(3000, 50)  # 50 goals every simulated season
    alloc = gm.allocate(team_goals, max_sims=3000)
    # total expected goals allocated ~ 50 (multinomial conserves the total)
    assert abs(alloc["exp_goals"].sum() - 50) < 1.0
    # a striker should out-score the goalkeeper
    gk = alloc[alloc.player == "Andreu Lladonosa"]["exp_goals"].iloc[0]
    st = alloc[alloc.player == "Adrià de Mesa"]["exp_goals"].iloc[0]
    assert st > gk
    # probabilities of being team top scorer sum to ~1
    assert abs(alloc["p_team_top_scorer"].sum() - 1.0) < 0.02


def test_penalty_taker_gets_boost():
    squad = _squad()
    pg = {"Adrià de Mesa": 10, "Chicho Barreda": 10, "Álex Rico": 4,
          "Andreu Lladonosa": 0}
    no_pen = GoalscorerModel(squad=squad, player_goals=pg, penalty_taker=None,
                             penalty_fraction=0.0).fit().allocate(np.full(2000, 50), max_sims=2000)
    with_pen = GoalscorerModel(squad=squad, player_goals=pg,
                               penalty_taker="Adrià de Mesa",
                               penalty_fraction=0.15).fit().allocate(np.full(2000, 50), max_sims=2000)
    a = lambda d: d[d.player == "Adrià de Mesa"]["exp_goals"].iloc[0]
    assert a(with_pen) > a(no_pen)


def test_anytime_probabilities_in_range():
    squad = _squad()
    pg = {"Adrià de Mesa": 14, "Chicho Barreda": 14, "Álex Rico": 4, "Andreu Lladonosa": 0}
    gm = GoalscorerModel(squad=squad, player_goals=pg).fit()
    at = gm.anytime_scorer_table(1.5)
    assert (at["p_anytime"] >= 0).all() and (at["p_anytime"] <= 1).all()


def test_league_pichichi_top_matches_best_scorer():
    teams = ["CD Binéfar", "CA Monzón"]
    gol = pd.DataFrame([
        {"player": "Youssef", "team": "CA Monzón", "goals": 32},
        {"player": "De Mesa", "team": "CD Binéfar", "goals": 14},
    ])
    # Monzón scores far more per season
    all_gf = np.vstack([np.full(2000, 45), np.full(2000, 75)])
    pich = league_top_scorer(all_gf, teams, gol, max_sims=2000)
    assert pich.iloc[0]["player_team"].startswith("Youssef")
    assert pich.iloc[0]["p_pichichi"] > 0.5
