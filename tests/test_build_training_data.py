# tests/test_build_training_data.py
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from src.cli import build_training_data
from src import config


@pytest.fixture
def fake_fetched_year_data(fake_fangraphs_df, fake_bref_df):
    """Returns a dict of (year -> (fg, bref, standings, awards))."""
    fg = fake_fangraphs_df.rename(columns={"WAR": "fWAR", "SO": "K",
                                            "xwOBA": "xwOBA_against", "RS/9": "RS_per_9"})
    bref = fake_bref_df.rename(columns={"Tm": "Team"})
    standings = pd.DataFrame({
        "Team": ["DET", "PIT", "OAK"], "year": [2024]*3,
        "team_winning_pct": [0.543, 0.469, 0.426],
    })
    awards = pd.DataFrame({
        "year": [2024], "league": ["AL"], "playerID": ["skubata01"],
        "pitcher_name": ["Tarik Skubal"], "pointsWon": [210],
        "vote_share": [1.0], "was_winner": [1], "award": ["Cy Young Award"],
    })
    fg["year"] = 2024
    bref["year"] = 2024
    return fg, bref, standings, awards


def test_build_training_data_writes_parquet(tmp_path, monkeypatch, fake_fetched_year_data):
    fg, bref, standings, awards = fake_fetched_year_data
    monkeypatch.setattr(config, "TRAINING_PARQUET", tmp_path / "training.parquet")
    monkeypatch.setattr(config, "TRAINING_YEARS", [2024])  # one year only for test

    with patch("src.cli.build_training_data.fetch.get_fangraphs_pitching", return_value=fg), \
         patch("src.cli.build_training_data.fetch.get_bref_pitching", return_value=bref), \
         patch("src.cli.build_training_data.fetch.get_team_records", return_value=standings), \
         patch("src.cli.build_training_data.fetch.get_awards_history", return_value=awards):
        build_training_data.main()

    out = pd.read_parquet(tmp_path / "training.parquet")
    assert len(out) > 0
    assert "vote_share" in out.columns
    for col in config.FEATURE_COLS:
        assert col in out.columns
