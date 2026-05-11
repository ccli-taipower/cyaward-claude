# tests/test_cli_daily.py
from datetime import date
from unittest.mock import patch
import pandas as pd
import pytest
from src.cli import daily


@pytest.fixture
def mock_ranking():
    rows = []
    for league in ["AL", "NL"]:
        for i in range(10):
            rows.append({
                "pitcher_name": f"P_{league}_{i+1}",
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


def test_daily_main_writes_predictions_and_site(tmp_path, monkeypatch, mock_ranking):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "predictions").mkdir(parents=True)
    (tmp_path / "site").mkdir(parents=True)
    (tmp_path / "templates").mkdir(parents=True)
    # Copy template into tmp path so render works
    import shutil
    from src.config import ROOT
    shutil.copy(ROOT / "templates" / "dashboard.html.j2",
                tmp_path / "templates" / "dashboard.html.j2")

    with patch("src.cli.daily.ranker.rank_today", return_value=mock_ranking), \
         patch("src.cli.daily.PREDICTIONS_DIR", tmp_path / "data" / "predictions"), \
         patch("src.cli.daily.SITE_DIR", tmp_path / "site"), \
         patch("src.render.TEMPLATES_DIR", tmp_path / "templates"):
        exit_code = daily.main(["--date", "2026-05-12"])

    assert exit_code == 0
    pred_path = tmp_path / "data" / "predictions" / "2026-05-12.parquet"
    site_path = tmp_path / "site" / "index.html"
    assert pred_path.exists()
    assert site_path.exists()
    pred = pd.read_parquet(pred_path)
    assert len(pred) == 20
