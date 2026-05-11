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
