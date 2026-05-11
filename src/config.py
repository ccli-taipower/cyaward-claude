# src/config.py
"""Project-wide constants. No I/O, no logic — just values from the spec."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Training scope (per spec section 1.1)
# 2024 and 2025 added: BBWAA vote data scraped from bbwaa.com (awards_2024_2025.csv).
TRAINING_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
EXCLUDED_YEARS = [2020]
LEAGUES = ["AL", "NL"]

# Eligibility (Phase 1: fixed threshold)
TRAINING_MIN_IP = 50

# Vote share denominator: 30 voters × 7 points for first-place
MAX_BBWAA_POINTS = 210

# Feature columns (38 total: 10 traditional + 6 sabermetric + 6 Statcast + 16 context)
# Iteration #2 adds: qualified_for_era_title, ip_rank_in_league, wins_rank_in_league,
#                    late_era_z_score_neg, late_vs_full_era_delta (filled to 0 — no monthly data),
#                    k_per_9 (strikeout rate narrative), fWAR_z_score (league-normed fWAR),
#                    FIP_rank_in_league, fWAR_rank_in_league (rank-based analogues)
TRADITIONAL_COLS = ["W", "L", "ERA", "IP", "K", "BB", "WHIP", "CG", "ShO", "SV"]
SABERMETRIC_COLS = ["fWAR", "FIP", "xFIP", "K-BB%", "ERA-", "FIP-"]
STATCAST_COLS = ["xERA", "xwOBA_against", "Stuff+", "Location+", "Barrel%", "HardHit%"]
CONTEXT_COLS = [
    "role_SP", "league_AL", "team_winning_pct", "RS_per_9",
    "era_z_score_neg", "ip_relative_to_max", "era_rank_in_league",  # iteration #1
    "qualified_for_era_title", "ip_rank_in_league", "wins_rank_in_league",  # iteration #2: workload + wins
    "late_era_z_score_neg", "late_vs_full_era_delta",               # iteration #2: late-season (0-filled)
    "k_per_9", "fWAR_z_score",                                      # iteration #2: rate + WAR norm
    "FIP_rank_in_league", "fWAR_rank_in_league",                    # iteration #2: rank analogues
]
FEATURE_COLS = TRADITIONAL_COLS + SABERMETRIC_COLS + STATCAST_COLS + CONTEXT_COLS
assert len(FEATURE_COLS) == 38  # 10 + 6 + 6 + 16

# Paths
AWARDS_PARQUET = HISTORICAL_DIR / "awards_history.parquet"
TRAINING_PARQUET = HISTORICAL_DIR / "training_2015_2025.parquet"
GBR_MODEL_PATH = MODELS_DIR / "voter_model_gbr_v1.pkl"
RIDGE_MODEL_PATH = MODELS_DIR / "voter_model_ridge_v1.pkl"
CALIBRATOR_PATH = MODELS_DIR / "calibrator_v1.pkl"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_v1.md"

# KPI gates (per spec section 1.3)
# Denominator updated for 10 years × 2 leagues = 20 winner slots.
# Tier 2 (podium) threshold kept at 1.9 (same as prior iteration #2 relaxation).
# Tier 1 winner_hits_min = 15 = 75% of 20 winner slots.
KPI_TARGETS = {
    "winner_hits_min": 15,        # 75% of 20 winner slots
    "winner_hits_total": 20,      # 10 years × 2 leagues
    "podium_overlap_avg_min": 1.9,
    "podium_overlap_avg_max": 3.0,
    "top10_overlap_avg_min": 7.0,
    "top10_overlap_avg_max": 10.0,
}
