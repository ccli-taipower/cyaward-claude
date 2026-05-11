# src/eligibility.py
"""Dynamic SP/RP IP eligibility thresholds for live mid-season ranking.

Per spec section 2.3: thresholds scale linearly with season progress.
Pre-season returns 0 progress (filter rejects everyone).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

SEASON_START = date(2026, 3, 26)
SEASON_END = date(2026, 9, 27)
SEASON_LENGTH_DAYS = (SEASON_END - SEASON_START).days  # 185

# Full-season IP floors (per spec)
SP_FULL_SEASON_IP = 162
RP_FULL_SEASON_IP = 60

# Minimum IP at any point in season (avoid empty rankings in early April)
SP_FLOOR_IP = 25
RP_FLOOR_IP = 10


def season_progress(today: date) -> float:
    """Fraction of regular season elapsed, clipped to [0, 1]."""
    if today < SEASON_START:
        return 0.0
    if today >= SEASON_END:
        return 1.0
    return (today - SEASON_START).days / SEASON_LENGTH_DAYS


def thresholds(today: date) -> tuple[float, float]:
    """Return (sp_min_ip, rp_min_ip) for ranking eligibility."""
    pct = season_progress(today)
    sp_min = max(SP_FLOOR_IP, SP_FULL_SEASON_IP * pct)
    rp_min = max(RP_FLOOR_IP, RP_FULL_SEASON_IP * pct)
    return sp_min, rp_min


def filter_eligible(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Filter a FanGraphs-shaped pitching frame to ranking-eligible pitchers.

    Pre-season -> empty DataFrame.
    Otherwise:
        SP (GS/G > 0.5) require IP >= sp_min
        RP (GS/G <= 0.5) require IP >= rp_min
    """
    if today < SEASON_START:
        return df.iloc[0:0].copy()
    sp_min, rp_min = thresholds(today)
    is_sp = df["GS"] / df["G"] > 0.5
    keep_sp = is_sp & (df["IP"] >= sp_min)
    keep_rp = (~is_sp) & (df["IP"] >= rp_min)
    return df[keep_sp | keep_rp].copy().reset_index(drop=True)
