# tests/test_render.py
from datetime import date
from pathlib import Path
import pandas as pd
import pytest
from src import render


@pytest.fixture
def sample_ranking():
    """20 pitchers, 10 per league, with all required schema columns."""
    rows = []
    for league in ["AL", "NL"]:
        for i in range(10):
            rows.append({
                "pitcher_name": f"Pitcher_{league}_{i+1}",
                "Team": "DET" if league == "AL" else "PIT",
                "league": league,
                "predicted_rank_in_league": i + 1,
                "predicted_vote_share": 0.5 - i * 0.04,
                "current_IP": 60.0 - i,
                "current_ERA": 2.0 + i * 0.1,
                "current_xERA": 2.2 + i * 0.1,
                "current_fWAR": 2.5 - i * 0.2,
                "proj_IP": 200.0 - i * 5,
                "proj_ERA": 2.0 + i * 0.1,
                "proj_fWAR": 6.0 - i * 0.4,
            })
    return pd.DataFrame(rows)


def test_render_writes_html_with_expected_content(sample_ranking, tmp_path):
    out_path = tmp_path / "index.html"
    render.render_dashboard(
        ranking=sample_ranking,
        asof_date=date(2026, 5, 12),
        out_path=out_path,
    )
    text = out_path.read_text()
    assert "MLB Cy Young Tracker" in text
    assert "American League" in text
    assert "National League" in text
    assert "Pitcher_AL_1" in text  # top AL pitcher
    assert "Pitcher_NL_1" in text  # top NL pitcher
    assert "2026-05-12" in text


def test_render_handles_empty_ranking(tmp_path):
    out_path = tmp_path / "index.html"
    render.render_dashboard(
        ranking=pd.DataFrame(columns=[
            "pitcher_name", "Team", "league", "predicted_rank_in_league",
            "predicted_vote_share", "current_IP", "current_ERA", "current_xERA",
            "current_fWAR", "proj_IP", "proj_ERA", "proj_fWAR",
        ]),
        asof_date=date(2026, 3, 1),
        out_path=out_path,
    )
    text = out_path.read_text()
    assert "MLB Cy Young Tracker" in text
    # Empty league section is OK — page still renders
