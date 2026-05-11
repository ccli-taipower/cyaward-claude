# src/features.py
"""Feature engineering: join raw frames, derive features, filter eligibility.

Pure functions (no I/O) — all inputs are DataFrames, output is DataFrame.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

from src.config import FEATURE_COLS, TRAINING_MIN_IP

# AL teams (used to infer league one-hot from team code)
AL_TEAMS = {"BAL", "BOS", "NYY", "TBR", "TOR",
            "CHW", "CLE", "DET", "KCR", "MIN",
            "HOU", "LAA", "OAK", "ATH", "SEA", "TEX"}


def _infer_league(team: str) -> str:
    return "AL" if team in AL_TEAMS else "NL"


def _normalize_name(name: str) -> str:
    """Strip accents / diacritics for fuzzy name matching.

    'Sandy Alcántara' -> 'Sandy Alcantara'
    """
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def build_features(
    year: int,
    fg: pd.DataFrame,
    bref: pd.DataFrame,
    standings: pd.DataFrame,
    awards: pd.DataFrame,
) -> pd.DataFrame:
    """Join raw frames into one training-row-per-pitcher DataFrame.

    - FG is the spine
    - Left-join standings (team -> winning pct)
    - Left-join awards (player -> vote_share); unmatched -> 0
    - Filter IP >= TRAINING_MIN_IP
    - Add derived: role_SP, league_AL, year tag

    Parameters
    ----------
    year:      Season year being processed.
    fg:        FanGraphs pitching stats (canonical column names already applied).
    bref:      Baseball-Reference pitching stats (reserved for v2; unused in MVP).
    standings: Pre-computed standings with columns [Team, team_winning_pct].
    awards:    Awards history with columns [year, pitcher_name, vote_share, was_winner].

    Returns
    -------
    DataFrame with columns [pitcher_name, Team, league, year] + FEATURE_COLS
    + [vote_share, was_winner], one row per eligible pitcher.
    """
    df = fg.copy()
    df = df[df["IP"] >= TRAINING_MIN_IP].copy()

    # Role one-hot (SP if started >50% of appearances)
    df["role_SP"] = (df["GS"] / df["G"] > 0.5).astype(int)

    # League one-hot — infer from Team
    df["league"] = df["Team"].map(_infer_league)
    df["league_AL"] = (df["league"] == "AL").astype(int)

    # Team winning pct
    df = df.merge(standings[["Team", "team_winning_pct"]], on="Team", how="left")

    # Awards label (left-join; unmatched = 0)
    # Normalize names to ASCII to handle accented characters (e.g. "Alcántara" -> "Alcantara")
    awards_slim = awards[awards["year"] == year][["pitcher_name", "vote_share", "was_winner"]].copy()
    awards_slim["pitcher_name_norm"] = awards_slim["pitcher_name"].map(_normalize_name)
    df["Name_norm"] = df["Name"].map(_normalize_name)
    df = df.merge(
        awards_slim.drop(columns=["pitcher_name"]),
        left_on="Name_norm", right_on="pitcher_name_norm",
        how="left",
    )
    df = df.drop(columns=["Name_norm", "pitcher_name_norm"], errors="ignore")
    # Restore pitcher_name for display
    awards_name_map = awards_slim.set_index("pitcher_name_norm")["pitcher_name"]
    df["pitcher_name"] = df["Name"].map(_normalize_name).map(awards_name_map)
    df["vote_share"] = df["vote_share"].fillna(0.0)
    df["was_winner"] = df["was_winner"].fillna(0).astype(int)

    # Standardize naming: pitcher_name may be NaN for unmatched rows
    if "pitcher_name" not in df.columns or df["pitcher_name"].isna().any():
        df["pitcher_name"] = df["pitcher_name"].fillna(df["Name"])
    df["year"] = year

    keep = ["pitcher_name", "Team", "league", "year"] + FEATURE_COLS + ["vote_share", "was_winner"]
    return df[keep].reset_index(drop=True)
