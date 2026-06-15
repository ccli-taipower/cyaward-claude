# tests/test_eligibility.py
from datetime import date
import pandas as pd
import pytest
from src import eligibility


@pytest.fixture
def sample_pitchers():
    """3 pitchers: SP with enough IP, SP with too few IP, RP with enough IP."""
    return pd.DataFrame({
        "Name": ["Skubal_SP_OK", "Rookie_SP_LOW", "Clase_RP_OK"],
        "G":    [10, 8, 18],
        "GS":   [10, 8, 0],     # GS/G > 0.5 → SP; GS/G == 0 → RP
        "IP":   [60.0, 12.0, 18.0],
    })


def test_season_progress_midseason():
    # 2026 season: 3/26 to 9/27 = ~185 days
    today = date(2026, 5, 12)  # 47 days in
    pct = eligibility.season_progress(today)
    assert 0.20 < pct < 0.30


def test_season_progress_pre_season_returns_zero():
    today = date(2026, 2, 1)
    assert eligibility.season_progress(today) == 0.0


def test_season_progress_post_season_capped_at_one():
    today = date(2026, 11, 1)
    assert eligibility.season_progress(today) == 1.0


def test_eligibility_thresholds_at_may_12():
    today = date(2026, 5, 12)
    sp_min, rp_min = eligibility.thresholds(today)
    # season_progress ≈ 0.254, sp_min = max(25, SP_FULL*0.254) — robust to constant change
    assert sp_min == pytest.approx(eligibility.SP_FULL_SEASON_IP * eligibility.season_progress(today))
    assert sp_min >= 25
    # rp_min = max(10, RP_FULL*0.254)
    assert rp_min == pytest.approx(eligibility.RP_FULL_SEASON_IP * eligibility.season_progress(today))
    assert rp_min >= 10


def test_filter_eligible_keeps_sp_above_threshold(sample_pitchers):
    today = date(2026, 5, 12)
    out = eligibility.filter_eligible(sample_pitchers, today)
    names = set(out["Name"])
    assert "Skubal_SP_OK" in names      # 60 IP > 41.1
    assert "Rookie_SP_LOW" not in names  # 12 IP < 41.1
    assert "Clase_RP_OK" in names        # RP, 18 IP > 15.2


def test_filter_eligible_returns_empty_pre_season(sample_pitchers):
    today = date(2026, 3, 1)  # before SEASON_START
    out = eligibility.filter_eligible(sample_pitchers, today)
    assert len(out) == 0
