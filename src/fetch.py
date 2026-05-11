# src/fetch.py
"""Wrappers around pybaseball + Lahman/B-Ref scrapers.

All wrappers normalize column names to a project-wide canonical set
(defined implicitly here) so downstream code never deals with raw
pybaseball naming quirks.

NOTE on blocked sources (as of 2025-05):
  - pybaseball.pitching_stats uses FanGraphs /leaders-legacy.aspx -> 403
  - pybaseball.pitching_stats_bref uses Baseball-Reference -> 403
  - pybaseball.standings uses Baseball-Reference -> 403
  - pybaseball.lahman uses chadwickbureau/baseballdatabank -> 404 (repo deleted)

Working alternatives used here:
  - FanGraphs JSON API (/api/leaders/major-league/data) for pitching stats
  - Baseball Savant CSV export for xwOBA against pitchers
  - MLB Stats API (statsapi.mlb.com) for team standings
  - jmaslek/LahmanDatabase mirror for AwardsSharePlayers
  - pybaseball.chadwick_register() still works
"""
from __future__ import annotations

import io
import time
import zipfile

import pandas as pd
import pybaseball as pyb
import requests

from src.config import MAX_BBWAA_POINTS, LEAGUES, HISTORICAL_DIR

# ---------------------------------------------------------------------------
# Lahman mirror (AwardsSharePlayers)
# ---------------------------------------------------------------------------
# pybaseball.lahman is broken (chadwickbureau/baseballdatabank deleted).
LAHMAN_MIRROR_URL = "https://github.com/jmaslek/LahmanDatabase/archive/refs/heads/main.zip"
LAHMAN_ZIP_CSV_PATH = "LahmanDatabase-main/unzipped/AwardsSharePlayers.csv"

# ---------------------------------------------------------------------------
# FanGraphs API (replacement for pybaseball.pitching_stats)
# ---------------------------------------------------------------------------
_FG_API_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
_FG_API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}
# FanGraphs API returns percentages as decimals (K-BB% = 0.29); convert to pct:
_FG_PCT_COLS = ["K-BB%"]
# FanGraphs API -> canonical name mapping
_FG_RENAME = {
    "WAR": "fWAR",
    "SO": "K",
    "RS/9": "RS_per_9",
    "sp_stuff": "Stuff+",
    "sp_location": "Location+",
}

# ---------------------------------------------------------------------------
# MLB Stats API (replacement for pybaseball.standings)
# ---------------------------------------------------------------------------
_MLB_STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"
_MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
# MLB API abbreviation -> FanGraphs abbreviation (for the non-matching ones)
_MLB_TO_FG_ABBR = {
    "AZ": "ARI",
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
    "CWS": "CHW",
    "ATH": "OAK",  # Sacramento Athletics (if they appear)
}

# ---------------------------------------------------------------------------
# Baseball Savant (for xwOBA against)
# ---------------------------------------------------------------------------
_SAVANT_EXPECTED_STATS_URL = (
    "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
)

pyb.cache.enable()  # use ~/.pybaseball cache for chadwick_register


def _fg_api_get(year: int) -> list[dict]:
    """Fetch raw JSON rows from FanGraphs API for a given year.

    Retries once after a 2-second pause if we get a 403 (Cloudflare may
    reset after a brief gap).
    """
    params = {
        "age": 0,
        "pos": "all",
        "stats": "pit",
        "lg": "all",
        "qual": 0,
        "season": year,
        "season1": year,
        "type": 8,
        "pageitems": 9999,
        "pagenum": 1,
    }
    for attempt in range(3):
        resp = requests.get(_FG_API_URL, params=params, headers=_FG_API_HEADERS, timeout=60)
        if resp.status_code == 200:
            return resp.json()["data"]
        if attempt < 2:
            wait = 3 * (attempt + 1)
            print(f"    FG API {year}: HTTP {resp.status_code}, retry in {wait}s ...")
            time.sleep(wait)
    resp.raise_for_status()


def _savant_get(year: int) -> pd.DataFrame:
    """Fetch xwOBA-against from Baseball Savant expected stats leaderboard."""
    params = {
        "type": "pitcher",
        "year": year,
        "position": "",
        "team": "",
        "min": 0,
        "csv": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    resp = requests.get(_SAVANT_EXPECTED_STATS_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content))
    return df


def get_fangraphs_pitching(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """All pitchers, FanGraphs leaderboard (via JSON API).

    Uses the FanGraphs /api/leaders/major-league/data endpoint (not the
    blocked /leaders-legacy.aspx).  Also fetches xwOBA-against from
    Baseball Savant and joins it via MLBAM player ID.

    Returns DataFrame with canonical columns plus a `year` column.

    Args:
        year: Season year to fetch.
        force_refresh: If True, skip the cache and re-fetch from the network
            (overwrites the cache file). Use for current-season calls where
            data changes daily. Historical years should use the default False.
    """
    cache_path = HISTORICAL_DIR / f"fg_pitching_{year}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    print(f"    Fetching FanGraphs pitching data for {year} ...")

    # --- FanGraphs API ---
    rows = _fg_api_get(year)
    fg = pd.DataFrame(rows)

    # Clean PlayerName (sometimes is HTML link in 'Name' col)
    fg["Name"] = fg["PlayerName"].fillna(fg["Name"])

    # Clean Team: use TeamNameAbb (clean FG abbr); 'Team' col has HTML links
    fg["Team"] = fg["TeamNameAbb"]

    # Rename columns to canonical names
    fg = fg.rename(columns=_FG_RENAME).copy()

    # K-BB% comes as decimal (0.291) -> convert to percentage (29.1)
    for col in _FG_PCT_COLS:
        if col in fg.columns:
            fg[col] = fg[col] * 100

    # --- Baseball Savant (xwOBA against) ---
    time.sleep(1)  # be polite between API calls
    try:
        savant = _savant_get(year)
        # savant has 'player_id' (MLBAM) and 'est_woba' (xwOBA against)
        savant_slim = savant[["player_id", "est_woba"]].rename(
            columns={"player_id": "xMLBAMID", "est_woba": "xwOBA_against"}
        )
        savant_slim["xMLBAMID"] = savant_slim["xMLBAMID"].astype("Int64")
        fg["xMLBAMID"] = pd.to_numeric(fg["xMLBAMID"], errors="coerce").astype("Int64")
        fg = fg.merge(savant_slim, on="xMLBAMID", how="left")
    except Exception as exc:
        print(f"    WARNING: Savant xwOBA fetch failed for {year}: {exc}")
        fg["xwOBA_against"] = float("nan")

    fg["year"] = year

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fg.to_csv(cache_path, index=False)
    print(f"    cached -> {cache_path}")

    return fg


def get_bref_pitching(year: int) -> pd.DataFrame:
    """Baseball-Reference pitching leaderboard — STUB (B-Ref is 403-blocked).

    bWAR is not used in MVP feature set (FEATURE_COLS uses fWAR from FG).
    Returns an empty DataFrame so build_features() still works.
    """
    # B-Ref (pybaseball.pitching_stats_bref) returns HTTP 403 as of 2025-05.
    # bWAR is reserved for Phase 2; the current feature set only uses fWAR.
    return pd.DataFrame(columns=["Name", "Team", "year", "bWAR"])


def get_team_records(year: int, force_refresh: bool = False) -> pd.DataFrame:
    """Final team standings -> winning pct per team.

    Uses the MLB Stats API (statsapi.mlb.com) instead of Baseball-Reference
    which is 403-blocked.  Returns team abbreviations in FanGraphs format
    so they join cleanly with get_fangraphs_pitching() output.

    Args:
        year: Season year to fetch.
        force_refresh: If True, skip the cache and re-fetch from the network
            (overwrites the cache file). Use for current-season calls where
            standings change daily. Historical years should use the default False.
    """
    cache_path = HISTORICAL_DIR / f"standings_{year}.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    # Fetch team abbreviations from MLB API
    teams_resp = requests.get(
        _MLB_TEAMS_URL,
        params={"sportId": 1, "season": year},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    teams_resp.raise_for_status()
    mlb_abbr = {
        t["id"]: _MLB_TO_FG_ABBR.get(t["abbreviation"], t["abbreviation"])
        for t in teams_resp.json()["teams"]
    }

    # Fetch standings
    standings_resp = requests.get(
        _MLB_STANDINGS_URL,
        params={
            "leagueId": "103,104",
            "season": year,
            "standingsTypes": "regularSeason",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    standings_resp.raise_for_status()

    parts = []
    for record in standings_resp.json()["records"]:
        for tr in record["teamRecords"]:
            team_id = tr["team"]["id"]
            team_abbr = mlb_abbr.get(team_id, tr["team"]["name"][:3].upper())
            w = tr["wins"]
            loss = tr["losses"]
            parts.append(
                {
                    "Team": team_abbr,
                    "year": year,
                    "team_winning_pct": w / (w + loss),
                }
            )

    df = pd.DataFrame(parts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _load_awards_share_players() -> pd.DataFrame:
    """Load AwardsSharePlayers from local cache; download from mirror if missing.

    pybaseball.lahman is broken (its hardcoded source repo was deleted).
    We use jmaslek/LahmanDatabase mirror as a workaround.
    Data covers through 2023; 2024 and 2025 are appended from BBWAA-scraped
    supplemental file (data/historical/awards_2024_2025.csv).
    """
    cache_path = HISTORICAL_DIR / "AwardsSharePlayers.csv"
    if not cache_path.exists():
        print("Downloading Lahman AwardsSharePlayers from mirror ...")
        resp = requests.get(LAHMAN_MIRROR_URL, stream=True, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            with zf.open(LAHMAN_ZIP_CSV_PATH) as f:
                df = pd.read_csv(f)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  cached -> {cache_path}")
    else:
        df = pd.read_csv(cache_path)

    # Append BBWAA-scraped 2024 + 2025 supplemental data
    supplemental_path = HISTORICAL_DIR / "awards_2024_2025.csv"
    if supplemental_path.exists():
        supp = pd.read_csv(supplemental_path)
        df = pd.concat([df, supp], ignore_index=True)
        print(f"  appended {len(supp)} rows from {supplemental_path.name}")

    return df


def get_awards_history(years: list[int]) -> pd.DataFrame:
    """Cy Young vote shares across requested years.

    Returns one row per (year, league, player) who received any votes,
    with computed `vote_share` (= pointsWon / 210) and `was_winner` (0/1).
    Player names are joined in via the Chadwick register.

    Note: jmaslek/LahmanDatabase mirror only has data through 2023.
    """
    raw = _load_awards_share_players()
    cy = raw[raw["awardID"] == "Cy Young Award"].copy()
    cy = cy[cy["yearID"].isin(years)].copy()
    cy = cy[cy["lgID"].isin(LEAGUES)].copy()

    cy["vote_share"] = cy["pointsWon"] / MAX_BBWAA_POINTS
    # was_winner: the player with the most pointsWon in each (year, league) group.
    # NOTE: pointsMax in Lahman is the theoretical maximum (always 210), NOT
    # the winner's actual points.  Using max() within group is the correct method.
    cy["max_points"] = cy.groupby(["yearID", "lgID"])["pointsWon"].transform("max")
    cy["was_winner"] = (cy["pointsWon"] == cy["max_points"]).astype(int)
    cy = cy.drop(columns=["max_points"])

    chadwick = pyb.chadwick_register()
    name_map = chadwick[["key_bbref", "name_first", "name_last"]].copy()
    name_map["pitcher_name"] = name_map["name_first"] + " " + name_map["name_last"]
    name_map = name_map[["key_bbref", "pitcher_name"]].rename(columns={"key_bbref": "playerID"})

    out = cy.merge(name_map, on="playerID", how="inner")
    out = out.rename(columns={"yearID": "year", "lgID": "league", "awardID": "award"})
    return out[["year", "league", "playerID", "pitcher_name", "pointsWon", "vote_share", "was_winner", "award"]]
