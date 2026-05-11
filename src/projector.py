# src/projector.py
"""Stage A: pace-based season projector.

Scales current cumulative stats (W, L, K, BB, IP, fWAR, ...) by the
inverse of season progress, leaving rate stats (ERA, FIP, xERA, ERA-,
FIP-, K-BB%, Stuff+, Location+, Barrel%, HardHit%, xwOBA_against, RS_per_9,
WHIP) unchanged.

Pre-season / post-season: pass through unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from src.eligibility import season_progress, SEASON_START, SEASON_END

# Stats that scale linearly with playing time
COUNTING_STATS = ["IP", "K", "BB", "W", "L", "fWAR", "CG", "ShO", "SV"]
# Stats that are already rates/ratios/indices — never scale them
RATE_STATS = ["ERA", "FIP", "xFIP", "xERA", "ERA-", "FIP-", "K-BB%",
              "WHIP", "Stuff+", "Location+", "Barrel%", "HardHit%",
              "xwOBA_against", "RS_per_9"]


class Projector(ABC):
    """Abstract interface (reserved for v3 Marcel-style projector)."""

    @abstractmethod
    def project(self, current: pd.DataFrame, asof_date: date) -> pd.DataFrame: ...


class PaceProjector(Projector):
    """Linear pace projection: IP_full = IP_current / season_progress."""

    def project(self, current: pd.DataFrame, asof_date: date) -> pd.DataFrame:
        # Pre-season or post-season: nothing to project
        if asof_date < SEASON_START or asof_date >= SEASON_END:
            return current.copy()

        pct = season_progress(asof_date)
        if pct <= 0:
            return current.copy()

        scale = 1.0 / pct  # e.g. 25% → multiply by 4
        out = current.copy()
        for col in COUNTING_STATS:
            if col in out.columns:
                out[col] = out[col] * scale
        # Rate stats: leave alone
        return out
