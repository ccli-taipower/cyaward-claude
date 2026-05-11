# tests/test_fetch.py
from unittest.mock import patch
import pandas as pd
import pytest
from src import fetch


def test_get_fangraphs_pitching_normalizes_columns(fake_fangraphs_df):
    with patch("src.fetch.pyb.pitching_stats", return_value=fake_fangraphs_df):
        out = fetch.get_fangraphs_pitching(2024)
    # Renaming check
    assert "fWAR" in out.columns        # WAR -> fWAR
    assert "K" in out.columns           # SO -> K
    assert "xwOBA_against" in out.columns  # xwOBA -> xwOBA_against
    assert "RS_per_9" in out.columns    # RS/9 -> RS_per_9
    # Year tag added
    assert (out["year"] == 2024).all()
    # Original rowcount preserved
    assert len(out) == 3


def test_get_bref_pitching_returns_bWAR(fake_bref_df):
    with patch("src.fetch.pyb.pitching_stats_bref", return_value=fake_bref_df):
        out = fetch.get_bref_pitching(2024)
    assert "bWAR" in out.columns
    assert "Name" in out.columns
    assert (out["year"] == 2024).all()


def test_get_team_records_returns_winning_pct(fake_standings_df):
    with patch("src.fetch.pyb.standings", return_value=fake_standings_df):
        out = fetch.get_team_records(2024)
    assert set(out.columns) == {"Team", "year", "team_winning_pct"}
    # DET: 88 / (88+74) = 0.5432...
    det = out[out["Team"] == "DET"]["team_winning_pct"].iloc[0]
    assert abs(det - 88 / 162) < 1e-3


def test_get_awards_history_filters_cy_young_only(fake_lahman_awards, fake_player_id_lookup):
    with patch("src.fetch.pyb.lahman.awards_share_players", return_value=fake_lahman_awards), \
         patch("src.fetch.pyb.chadwick_register", return_value=fake_player_id_lookup):
        out = fetch.get_awards_history([2023, 2024])
    assert (out["award"] == "Cy Young Award").all()
    assert set(out["year"].unique()) == {2023, 2024}


def test_awards_vote_share_normalized(fake_lahman_awards, fake_player_id_lookup):
    with patch("src.fetch.pyb.lahman.awards_share_players", return_value=fake_lahman_awards), \
         patch("src.fetch.pyb.chadwick_register", return_value=fake_player_id_lookup):
        out = fetch.get_awards_history([2023, 2024])
    skubal = out[out["playerID"] == "skubata01"].iloc[0]
    assert skubal["vote_share"] == pytest.approx(1.0)  # 210/210 unanimous
    assert skubal["was_winner"] == 1
    ragans = out[out["playerID"] == "ragansh01"].iloc[0]
    assert ragans["vote_share"] == pytest.approx(95 / 210)
    assert ragans["was_winner"] == 0
    # Player Name attached via Chadwick lookup
    assert skubal["pitcher_name"] == "Tarik Skubal"


def test_awards_handles_missing_year_gracefully(fake_lahman_awards, fake_player_id_lookup):
    # Request 2024 only — should return only 2024 rows
    with patch("src.fetch.pyb.lahman.awards_share_players", return_value=fake_lahman_awards), \
         patch("src.fetch.pyb.chadwick_register", return_value=fake_player_id_lookup):
        out = fetch.get_awards_history([2024])
    assert (out["year"] == 2024).all()
    assert len(out) == 4
