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
    fg_late_season: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join raw frames into one training-row-per-pitcher DataFrame.

    - FG is the spine
    - Left-join standings (team -> winning pct)
    - Left-join awards (player -> vote_share); unmatched -> 0
    - Filter IP >= TRAINING_MIN_IP
    - Add derived: role_SP, league_AL, year tag

    Parameters
    ----------
    year:           Season year being processed.
    fg:             FanGraphs pitching stats (canonical column names already applied).
    bref:           Baseball-Reference pitching stats (reserved for v2; unused in MVP).
    standings:      Pre-computed standings with columns [Team, team_winning_pct].
    awards:         Awards history with columns [year, pitcher_name, vote_share, was_winner].
    fg_late_season: Optional late-season (Aug+Sep) FanGraphs data.  When provided,
                    enables late_era_z_score_neg and late_vs_full_era_delta features.
                    When None or empty, those features are filled with 0.0.

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

    # Compute league-context features (within this year, grouped by league)
    era_grp = df.groupby("league")["ERA"]
    df["era_z_score_neg"] = -((df["ERA"] - era_grp.transform("mean")) / era_grp.transform("std"))
    ip_grp = df.groupby("league")["IP"]
    df["ip_relative_to_max"] = df["IP"] / ip_grp.transform("max")
    df["era_rank_in_league"] = era_grp.rank(method="min", ascending=True)

    # Iteration #2: workload features
    # Hard threshold: ERA title requires 162 IP (1 IP per scheduled game)
    df["qualified_for_era_title"] = (df["IP"] >= 162).astype(int)
    # IP rank within league (1 = most innings; workhorse signal)
    df["ip_rank_in_league"] = ip_grp.rank(method="min", ascending=False)
    # Wins rank within league (1 = most wins; captures narrative like Porcello 2016)
    df["wins_rank_in_league"] = df.groupby("league")["W"].rank(method="min", ascending=False)

    # Iteration #2: late-season features
    # FanGraphs monthly split API is Cloudflare-blocked (returns 403).
    # When fg_late_season is provided (future: alternative source), compute real values.
    # For now, fill with 0.0 so the feature exists and the model can learn around it.
    if fg_late_season is not None and len(fg_late_season) > 0:
        late = fg_late_season[["Name", "ERA", "IP"]].rename(
            columns={"ERA": "late_ERA", "IP": "late_IP"}
        )
        df = df.merge(late, on="Name", how="left")
        late_grp = df.groupby("league")["late_ERA"]
        df["late_era_z_score_neg"] = -(
            (df["late_ERA"] - late_grp.transform("mean")) / late_grp.transform("std")
        )
        df["late_vs_full_era_delta"] = df["late_ERA"] - df["ERA"]
    else:
        df["late_era_z_score_neg"] = 0.0
        df["late_vs_full_era_delta"] = 0.0

    # Iteration #2: rate and normalized WAR features
    # K/9 as a rate stat (strikeout narrative — e.g. Burnes 2021 historic K rate)
    df["k_per_9"] = df["K"] / df["IP"] * 9
    # fWAR z-score within league: normalizes fWAR so that a high fWAR in a weak
    # league year is discounted vs. a dominant fWAR season.
    fwar_grp = df.groupby("league")["fWAR"]
    df["fWAR_z_score"] = (df["fWAR"] - fwar_grp.transform("mean")) / fwar_grp.transform("std")
    # Rank-based analogues for FIP and fWAR to give the model ordinal signals.
    # FIP rank (1 = best FIP in league; captures ace-level dominance like Burnes 2021)
    df["FIP_rank_in_league"] = df.groupby("league")["FIP"].rank(method="min", ascending=True)
    # fWAR rank (1 = highest fWAR in league)
    df["fWAR_rank_in_league"] = fwar_grp.rank(method="min", ascending=False)

    keep = ["pitcher_name", "Team", "league", "year"] + FEATURE_COLS + ["vote_share", "was_winner"]
    return df[keep].reset_index(drop=True)
