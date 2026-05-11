# tests/test_projector.py
from datetime import date
import pandas as pd
import pytest
from src import projector


@pytest.fixture
def current_stats_midseason():
    """A pitcher with ~40 IP at season_progress=0.25 (should project to ~160 IP)."""
    return pd.DataFrame({
        "Name":    ["Skubal", "Skenes"],
        "Team":    ["DET", "PIT"],
        "G":       [8, 7],
        "GS":      [8, 7],
        "IP":      [50.0, 40.0],
        "K":       [60, 55],
        "BB":      [10, 8],
        "ERA":     [2.50, 1.80],
        "fWAR":    [1.5, 1.2],
        "W":       [4, 3],
        "L":       [1, 2],
        "WHIP":    [0.95, 0.88],
        "CG":      [0, 0],
        "ShO":     [0, 0],
        "SV":      [0, 0],
        "FIP":     [2.40, 2.10],
        "xFIP":    [2.80, 2.50],
        "K-BB%":   [25.0, 24.5],
        "ERA-":    [60, 45],
        "FIP-":    [62, 55],
        "xERA":    [2.85, 2.20],
        "xwOBA_against": [0.255, 0.240],
        "Stuff+":  [115, 130],
        "Location+": [105, 102],
        "Barrel%": [5.5, 4.0],
        "HardHit%": [33.0, 30.0],
        "RS_per_9": [4.8, 5.2],
    })


def test_project_scales_ip_to_full_season(current_stats_midseason):
    today = date(2026, 5, 12)  # ~25% of season
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    # 50 IP at 25% → ~200 IP projected
    skubal_proj = proj[proj["Name"] == "Skubal"].iloc[0]
    assert 150 < skubal_proj["IP"] < 250


def test_project_preserves_rate_stats(current_stats_midseason):
    today = date(2026, 5, 12)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    skubal_orig = current_stats_midseason[current_stats_midseason["Name"] == "Skubal"].iloc[0]
    skubal_proj = proj[proj["Name"] == "Skubal"].iloc[0]
    # ERA, FIP, xERA, ERA-, Stuff+ are rate/index stats — should NOT scale
    assert skubal_proj["ERA"] == pytest.approx(skubal_orig["ERA"])
    assert skubal_proj["xERA"] == pytest.approx(skubal_orig["xERA"])
    assert skubal_proj["Stuff+"] == pytest.approx(skubal_orig["Stuff+"])


def test_project_scales_counting_stats(current_stats_midseason):
    today = date(2026, 5, 12)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    skubal_orig = current_stats_midseason[current_stats_midseason["Name"] == "Skubal"].iloc[0]
    skubal_proj = proj[proj["Name"] == "Skubal"].iloc[0]
    # K, BB, W, L, fWAR should scale up with IP
    ratio = skubal_proj["IP"] / skubal_orig["IP"]
    assert skubal_proj["K"] == pytest.approx(skubal_orig["K"] * ratio, rel=0.01)
    assert skubal_proj["fWAR"] == pytest.approx(skubal_orig["fWAR"] * ratio, rel=0.01)
    assert skubal_proj["W"] == pytest.approx(skubal_orig["W"] * ratio, rel=0.01)


def test_project_pre_season_returns_inputs_unchanged(current_stats_midseason):
    today = date(2026, 2, 1)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    # Pre-season: nothing to project, just pass through
    pd.testing.assert_frame_equal(
        proj.reset_index(drop=True),
        current_stats_midseason.reset_index(drop=True),
        check_dtype=False,
    )


def test_project_post_season_returns_inputs_unchanged(current_stats_midseason):
    today = date(2026, 11, 1)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    pd.testing.assert_frame_equal(
        proj.reset_index(drop=True),
        current_stats_midseason.reset_index(drop=True),
        check_dtype=False,
    )
