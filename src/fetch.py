# src/fetch.py
"""Wrappers around pybaseball + Lahman/B-Ref scrapers.

All wrappers normalize column names to a project-wide canonical set
(defined implicitly here) so downstream code never deals with raw
pybaseball naming quirks.
"""
from __future__ import annotations

import pandas as pd
import pybaseball as pyb

pyb.cache.enable()  # use ~/.pybaseball cache

_FG_RENAME = {
    "WAR": "fWAR",
    "SO": "K",
    "xwOBA": "xwOBA_against",
    "RS/9": "RS_per_9",
}


def get_fangraphs_pitching(year: int) -> pd.DataFrame:
    """All pitchers, FanGraphs leaderboard. qual=0 keeps reliever-only IP totals.

    Returns DataFrame with canonical columns (renamed per _FG_RENAME) plus a
    `year` column.
    """
    df = pyb.pitching_stats(year, year, qual=0)
    df = df.rename(columns=_FG_RENAME).copy()
    df["year"] = year
    return df


def get_bref_pitching(year: int) -> pd.DataFrame:
    """Baseball-Reference pitching leaderboard (for bWAR, holds, etc.)."""
    df = pyb.pitching_stats_bref(year)
    df = df.copy()
    df["year"] = year
    # Standardize team col name with FG ('Tm' -> 'Team')
    if "Tm" in df.columns:
        df = df.rename(columns={"Tm": "Team"})
    return df


def get_team_records(year: int) -> pd.DataFrame:
    """Final team standings -> winning pct per team.

    pybaseball.standings(year) returns list[DataFrame] (one per division).
    """
    divisions = pyb.standings(year)
    parts = []
    for div in divisions:
        d = div.copy()
        # Some pybaseball versions name the team col 'Tm', others 'Team'
        if "Tm" in d.columns and "Team" not in d.columns:
            d = d.rename(columns={"Tm": "Team"})
        d["W"] = d["W"].astype(int)
        d["L"] = d["L"].astype(int)
        d["team_winning_pct"] = d["W"] / (d["W"] + d["L"])
        d["year"] = year
        parts.append(d[["Team", "year", "team_winning_pct"]])
    return pd.concat(parts, ignore_index=True)
