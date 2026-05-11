# tests/test_fetch.py
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from src import fetch


# ---------------------------------------------------------------------------
# FanGraphs pitching tests (now uses FG JSON API, not pyb.pitching_stats)
# ---------------------------------------------------------------------------

def _make_fg_api_rows(year: int = 2023) -> list[dict]:
    """Fake FG API rows matching the shape the real API returns."""
    return [
        {
            "PlayerName": "Tarik Skubal", "TeamNameAbb": "DET",
            "WAR": 5.5, "SO": 228, "BB": 35, "W": 18, "L": 4,
            "ERA": 2.39, "IP": 192.0, "WHIP": 0.92, "CG": 1, "ShO": 0, "SV": 0,
            "FIP": 2.49, "xFIP": 2.93, "K-BB%": 0.255, "ERA-": 60.0, "FIP-": 62.0,
            "xERA": 2.85, "sp_stuff": 115.0, "sp_location": 105.0,
            "Barrel%": 0.055, "HardHit%": 0.330, "RS/9": 4.8,
            "xMLBAMID": 669373, "playerid": 12345, "G": 31, "GS": 31, "Name": "Tarik Skubal",
        },
        {
            "PlayerName": "Paul Skenes", "TeamNameAbb": "PIT",
            "WAR": 4.3, "SO": 170, "BB": 32, "W": 11, "L": 3,
            "ERA": 1.96, "IP": 133.0, "WHIP": 0.95, "CG": 0, "ShO": 0, "SV": 0,
            "FIP": 2.44, "xFIP": 2.83, "K-BB%": 0.238, "ERA-": 49.0, "FIP-": 60.0,
            "xERA": 2.20, "sp_stuff": 130.0, "sp_location": 102.0,
            "Barrel%": 0.040, "HardHit%": 0.300, "RS/9": 5.2,
            "xMLBAMID": 694973, "playerid": 23456, "G": 23, "GS": 23, "Name": "Paul Skenes",
        },
    ]


def _make_savant_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [669373, 694973],
        "est_woba": [0.255, 0.240],
    })


def test_get_fangraphs_pitching_normalizes_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "HISTORICAL_DIR", tmp_path)
    with patch("src.fetch._fg_api_get", return_value=_make_fg_api_rows()), \
         patch("src.fetch._savant_get", return_value=_make_savant_df()):
        out = fetch.get_fangraphs_pitching(2023)
    # Renaming check
    assert "fWAR" in out.columns          # WAR -> fWAR
    assert "K" in out.columns             # SO -> K
    assert "xwOBA_against" in out.columns # from Savant join
    assert "RS_per_9" in out.columns      # RS/9 -> RS_per_9
    assert "Stuff+" in out.columns        # sp_stuff -> Stuff+
    assert "Location+" in out.columns     # sp_location -> Location+
    # Year tag added
    assert (out["year"] == 2023).all()
    # Original rowcount preserved
    assert len(out) == 2


def test_get_fangraphs_pitching_kbb_pct_in_percent(tmp_path, monkeypatch):
    """FG API returns K-BB% as decimal (0.255); we convert to percentage (25.5)."""
    monkeypatch.setattr(fetch, "HISTORICAL_DIR", tmp_path)
    with patch("src.fetch._fg_api_get", return_value=_make_fg_api_rows()), \
         patch("src.fetch._savant_get", return_value=_make_savant_df()):
        out = fetch.get_fangraphs_pitching(2023)
    skubal = out[out["Name"] == "Tarik Skubal"].iloc[0]
    assert abs(skubal["K-BB%"] - 25.5) < 0.01


def test_get_bref_pitching_returns_stub():
    """get_bref_pitching is a stub (B-Ref is blocked); should return empty DF."""
    out = fetch.get_bref_pitching(2023)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    assert "bWAR" in out.columns


def test_get_team_records_returns_winning_pct(tmp_path, monkeypatch):
    """get_team_records now uses MLB Stats API; mock both sub-calls."""
    monkeypatch.setattr(fetch, "HISTORICAL_DIR", tmp_path)

    fake_teams_resp = MagicMock()
    fake_teams_resp.raise_for_status = MagicMock()
    fake_teams_resp.json.return_value = {
        "teams": [
            {"id": 116, "abbreviation": "DET"},
            {"id": 134, "abbreviation": "PIT"},
        ]
    }

    fake_standings_resp = MagicMock()
    fake_standings_resp.raise_for_status = MagicMock()
    fake_standings_resp.json.return_value = {
        "records": [
            {"teamRecords": [
                {"team": {"id": 116, "name": "Detroit Tigers"}, "wins": 88, "losses": 74},
                {"team": {"id": 134, "name": "Pittsburgh Pirates"}, "wins": 76, "losses": 86},
            ]}
        ]
    }

    responses = [fake_teams_resp, fake_standings_resp]
    with patch("src.fetch.requests.get", side_effect=responses):
        out = fetch.get_team_records(2023)

    assert set(out.columns) == {"Team", "year", "team_winning_pct"}
    det = out[out["Team"] == "DET"]["team_winning_pct"].iloc[0]
    assert abs(det - 88 / 162) < 1e-3


# ---------------------------------------------------------------------------
# Awards tests (mock _load_awards_share_players, unchanged logic)
# ---------------------------------------------------------------------------

def test_get_awards_history_filters_cy_young_only(fake_lahman_awards, fake_player_id_lookup):
    with patch("src.fetch._load_awards_share_players", return_value=fake_lahman_awards), \
         patch("src.fetch.pyb.chadwick_register", return_value=fake_player_id_lookup):
        out = fetch.get_awards_history([2023, 2024])
    assert (out["award"] == "Cy Young Award").all()
    assert set(out["year"].unique()) == {2023, 2024}


def test_awards_vote_share_normalized(fake_lahman_awards, fake_player_id_lookup):
    with patch("src.fetch._load_awards_share_players", return_value=fake_lahman_awards), \
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
    with patch("src.fetch._load_awards_share_players", return_value=fake_lahman_awards), \
         patch("src.fetch.pyb.chadwick_register", return_value=fake_player_id_lookup):
        out = fetch.get_awards_history([2024])
    assert (out["year"] == 2024).all()
    assert len(out) == 4
