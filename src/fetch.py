# src/fetch.py
"""Wrappers around pybaseball + Lahman/B-Ref scrapers.

All wrappers normalize column names to a project-wide canonical set
(defined implicitly here) so downstream code never deals with raw
pybaseball naming quirks.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pybaseball as pyb
import requests

from src.config import MAX_BBWAA_POINTS, LEAGUES, HISTORICAL_DIR

# pybaseball.lahman is broken (its hardcoded source repo was deleted).
# We use jmaslek/LahmanDatabase mirror as a workaround.
LAHMAN_MIRROR_URL = "https://github.com/jmaslek/LahmanDatabase/archive/refs/heads/main.zip"
LAHMAN_ZIP_CSV_PATH = "LahmanDatabase-main/unzipped/AwardsSharePlayers.csv"

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


def _load_awards_share_players() -> pd.DataFrame:
    """Load AwardsSharePlayers from local cache; download from mirror if missing.

    pybaseball.lahman is broken (its hardcoded source repo was deleted).
    We use jmaslek/LahmanDatabase mirror as a workaround.
    Data covers through 2023; 2024 is not yet available in this mirror.
    """
    cache_path = HISTORICAL_DIR / "AwardsSharePlayers.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    print("Downloading Lahman AwardsSharePlayers from mirror ...")
    resp = requests.get(LAHMAN_MIRROR_URL, stream=True, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(LAHMAN_ZIP_CSV_PATH) as f:
            df = pd.read_csv(f)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  cached -> {cache_path}")
    return df


def get_awards_history(years: list[int]) -> pd.DataFrame:
    """Cy Young vote shares across requested years.

    Returns one row per (year, league, player) who received any votes,
    with computed `vote_share` (= pointsWon / 210) and `was_winner` (0/1).
    Player names are joined in via the Chadwick register.

    Note: pybaseball's bundled Lahman snapshot may lag the current season
    by ~6 months. Caller must verify all expected years are present
    (see Task 5 build_training_data.py's verify step).
    """
    raw = _load_awards_share_players()
    cy = raw[raw["awardID"] == "Cy Young Award"].copy()
    cy = cy[cy["yearID"].isin(years)].copy()
    cy = cy[cy["lgID"].isin(LEAGUES)].copy()

    cy["vote_share"] = cy["pointsWon"] / MAX_BBWAA_POINTS
    cy["was_winner"] = (cy["pointsWon"] == cy["pointsMax"]).astype(int)

    chadwick = pyb.chadwick_register()
    name_map = chadwick[["key_bbref", "name_first", "name_last"]].copy()
    name_map["pitcher_name"] = name_map["name_first"] + " " + name_map["name_last"]
    name_map = name_map[["key_bbref", "pitcher_name"]].rename(columns={"key_bbref": "playerID"})

    out = cy.merge(name_map, on="playerID", how="inner")
    out = out.rename(columns={"yearID": "year", "lgID": "league", "awardID": "award"})
    return out[["year", "league", "playerID", "pitcher_name", "pointsWon", "vote_share", "was_winner", "award"]]
