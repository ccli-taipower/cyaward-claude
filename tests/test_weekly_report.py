# tests/test_weekly_report.py
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import pytest
from src import weekly_report


@pytest.fixture
def fake_predictions_week(tmp_path):
    """Write 7 days of fake prediction parquets, with one pitcher rising and one falling."""
    pred_dir = tmp_path / "data" / "predictions"
    pred_dir.mkdir(parents=True)
    base = date(2026, 5, 5)  # Tuesday of week 19
    for i in range(7):
        day = base + timedelta(days=i)
        rows = []
        for league in ["AL", "NL"]:
            for j in range(10):
                # "Rising" pitcher (AL_riser): starts rank 8, climbs to rank 2 over the week
                # "Falling" pitcher (NL_faller): starts rank 1, falls to rank 5
                if league == "AL" and j == 1:
                    name = "AL_riser"
                    rank = 8 - i  # 8 → 2
                elif league == "NL" and j == 0:
                    name = "NL_faller"
                    rank = 1 + i // 2  # 1 → 4
                else:
                    name = f"P_{league}_{j+1}"
                    rank = j + 1
                rows.append({
                    "pitcher_name": name,
                    "Team": "DET",
                    "league": league,
                    "predicted_rank_in_league": rank,
                    "predicted_vote_share": 0.6 - rank * 0.05,
                    "current_IP": 60.0, "current_ERA": 2.5,
                    "current_xERA": 2.5, "current_fWAR": 2.0,
                    "proj_IP": 200.0, "proj_ERA": 2.5, "proj_fWAR": 6.0,
                })
        pd.DataFrame(rows).to_parquet(pred_dir / f"{day.isoformat()}.parquet", index=False)
    return pred_dir


def test_weekly_report_generates_markdown_with_sections(fake_predictions_week, tmp_path):
    out_path = tmp_path / "reports" / "2026-W19.md"
    weekly_report.generate_weekly_report(
        predictions_dir=fake_predictions_week,
        out_path=out_path,
        week_end=date(2026, 5, 11),
    )
    text = out_path.read_text()
    assert "Cy Young Weekly" in text
    assert "AL" in text
    assert "NL" in text
    assert "Top 10" in text


def test_weekly_report_flags_rising_and_falling(fake_predictions_week, tmp_path):
    out_path = tmp_path / "reports" / "2026-W19.md"
    weekly_report.generate_weekly_report(
        predictions_dir=fake_predictions_week,
        out_path=out_path,
        week_end=date(2026, 5, 11),
    )
    text = out_path.read_text()
    # Rising pitcher (rank moved 8 → 2, +6) should appear
    assert "AL_riser" in text
    # Falling pitcher (rank moved 1 → 4, -3) should appear
    assert "NL_faller" in text
