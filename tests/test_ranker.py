# tests/test_ranker.py
from datetime import date
from unittest.mock import patch
import pandas as pd
import pytest
import numpy as np
from src import ranker, config


@pytest.fixture
def mock_current_data(fake_fangraphs_df):
    """Reuse fake_fangraphs_df from conftest; rename to canonical names."""
    df = fake_fangraphs_df.rename(columns={
        "WAR": "fWAR", "SO": "K", "xwOBA": "xwOBA_against", "RS/9": "RS_per_9"
    }).copy()
    df["year"] = 2026
    # Bump IP to pass mid-season threshold
    df.loc[:, "IP"] = [80.0, 75.0, 70.0]
    df.loc[:, "G"] = [12, 11, 10]
    df.loc[:, "GS"] = [12, 11, 10]
    return df


@pytest.fixture
def mock_standings_2026():
    return pd.DataFrame({
        "Team": ["DET", "PIT", "OAK"],
        "year": [2026, 2026, 2026],
        "team_winning_pct": [0.580, 0.460, 0.420],
    })


def test_rank_today_returns_expected_schema(mock_current_data, mock_standings_2026):
    today = date(2026, 5, 12)
    with patch("src.ranker.fetch.get_fangraphs_pitching", return_value=mock_current_data), \
         patch("src.ranker.fetch.get_team_records", return_value=mock_standings_2026), \
         patch("src.ranker.fetch.get_bref_pitching", return_value=pd.DataFrame(columns=["Name","Team","year","bWAR"])):
        result = ranker.rank_today(today)
    # Required columns
    assert "pitcher_name" in result.columns
    assert "Team" in result.columns
    assert "league" in result.columns
    assert "predicted_vote_share" in result.columns
    assert "predicted_rank_in_league" in result.columns
    assert "current_IP" in result.columns
    assert "current_ERA" in result.columns
    assert "current_fWAR" in result.columns
    assert "proj_IP" in result.columns


def test_rank_today_sorts_descending_within_league(mock_current_data, mock_standings_2026):
    today = date(2026, 5, 12)
    with patch("src.ranker.fetch.get_fangraphs_pitching", return_value=mock_current_data), \
         patch("src.ranker.fetch.get_team_records", return_value=mock_standings_2026), \
         patch("src.ranker.fetch.get_bref_pitching", return_value=pd.DataFrame(columns=["Name","Team","year","bWAR"])):
        result = ranker.rank_today(today)
    for league, grp in result.groupby("league"):
        shares = grp.sort_values("predicted_rank_in_league")["predicted_vote_share"].values
        # rank 1 should have highest predicted share within league
        assert all(shares[i] >= shares[i+1] for i in range(len(shares)-1))


def test_rank_today_pre_season_returns_empty(mock_current_data, mock_standings_2026):
    today = date(2026, 2, 1)
    with patch("src.ranker.fetch.get_fangraphs_pitching", return_value=mock_current_data), \
         patch("src.ranker.fetch.get_team_records", return_value=mock_standings_2026), \
         patch("src.ranker.fetch.get_bref_pitching", return_value=pd.DataFrame(columns=["Name","Team","year","bWAR"])):
        result = ranker.rank_today(today)
    assert len(result) == 0
