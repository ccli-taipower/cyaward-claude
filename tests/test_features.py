# tests/test_features.py
import pandas as pd
import pytest
from src import features


@pytest.fixture
def joined_inputs(fake_fangraphs_df, fake_bref_df, fake_standings_df, fake_lahman_awards, fake_player_id_lookup):
    """Build the inputs that build_features expects, using fixtures."""
    fg = fake_fangraphs_df.copy()
    fg = fg.rename(columns={"WAR": "fWAR", "SO": "K", "xwOBA": "xwOBA_against", "RS/9": "RS_per_9"})
    fg["year"] = 2024
    bref = fake_bref_df.copy()
    bref["year"] = 2024
    bref = bref.rename(columns={"Tm": "Team"})
    standings = pd.DataFrame({
        "Team": ["DET", "PIT", "OAK"],
        "year": [2024, 2024, 2024],
        "team_winning_pct": [0.543, 0.469, 0.426],
    })
    awards = pd.DataFrame({
        "year": [2024, 2024],
        "league": ["AL", "NL"],
        "playerID": ["skubata01", "skenepa01"],
        "pitcher_name": ["Tarik Skubal", "Paul Skenes"],
        "pointsWon": [210, 90],
        "vote_share": [1.0, 90/210],
        "was_winner": [1, 0],
        "award": ["Cy Young Award", "Cy Young Award"],
    })
    return fg, bref, standings, awards


def test_build_features_returns_canonical_columns(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    out = features.build_features(2024, fg, bref, standings, awards)
    from src.config import FEATURE_COLS
    for col in FEATURE_COLS:
        assert col in out.columns, f"missing feature: {col}"
    assert "vote_share" in out.columns
    assert "pitcher_name" in out.columns
    assert "league" in out.columns


def test_build_features_filters_by_min_ip(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    fg.loc[fg["Name"] == "Bad Pitcher", "IP"] = 10  # below threshold
    out = features.build_features(2024, fg, bref, standings, awards)
    assert "Bad Pitcher" not in out["pitcher_name"].values


def test_build_features_unmatched_pitchers_get_zero_vote_share(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    out = features.build_features(2024, fg, bref, standings, awards)
    bad = out[out["pitcher_name"] == "Bad Pitcher"]
    assert len(bad) == 1
    assert bad["vote_share"].iloc[0] == 0.0
    assert bad["was_winner"].iloc[0] == 0


def test_build_features_role_one_hot(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    out = features.build_features(2024, fg, bref, standings, awards)
    # All three fixture pitchers are SP (G == GS)
    assert (out["role_SP"] == 1).all()


def test_build_features_league_one_hot_inferred(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    out = features.build_features(2024, fg, bref, standings, awards)
    skubal = out[out["pitcher_name"] == "Tarik Skubal"].iloc[0]
    assert skubal["league_AL"] == 1
    skenes = out[out["pitcher_name"] == "Paul Skenes"].iloc[0]
    assert skenes["league_AL"] == 0


def test_build_features_era_zscore_winner_has_positive_value(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    out = features.build_features(2024, fg, bref, standings, awards)
    # Skubal ERA=2.39 (AL), Bad ERA=5.50 (AL) -> mean ~3.94, Skubal z-neg should be POSITIVE
    skubal = out[out["pitcher_name"] == "Tarik Skubal"].iloc[0]
    bad = out[out["pitcher_name"] == "Bad Pitcher"].iloc[0]
    # AL has 2 pitchers: Skubal 2.39, Bad 5.50 -> mean ~3.94, Skubal z-neg should be POSITIVE
    assert skubal["era_z_score_neg"] > 0  # better than mean
    assert bad["era_z_score_neg"] < 0      # worse than mean


def test_build_features_ip_relative_workload(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    out = features.build_features(2024, fg, bref, standings, awards)
    # AL: Skubal 192 IP, Bad 158 IP. Max = 192. So Skubal=1.0, Bad ~= 0.823
    skubal = out[out["pitcher_name"] == "Tarik Skubal"].iloc[0]
    bad = out[out["pitcher_name"] == "Bad Pitcher"].iloc[0]
    assert skubal["ip_relative_to_max"] == pytest.approx(1.0)
    assert bad["ip_relative_to_max"] == pytest.approx(158 / 192, abs=0.01)


def test_build_features_era_rank_within_league(joined_inputs):
    fg, bref, standings, awards = joined_inputs
    out = features.build_features(2024, fg, bref, standings, awards)
    # AL: Skubal 2.39 (rank 1), Bad 5.50 (rank 2). NL: Skenes 1.96 (rank 1, alone).
    skubal = out[out["pitcher_name"] == "Tarik Skubal"].iloc[0]
    bad = out[out["pitcher_name"] == "Bad Pitcher"].iloc[0]
    skenes = out[out["pitcher_name"] == "Paul Skenes"].iloc[0]
    assert skubal["era_rank_in_league"] == 1
    assert bad["era_rank_in_league"] == 2
    assert skenes["era_rank_in_league"] == 1
