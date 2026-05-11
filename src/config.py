# src/config.py
"""Project-wide constants. No I/O, no logic — just values from the spec."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Training scope (per spec section 1.1)
# 2024 excluded: Lahman mirror (jmaslek/LahmanDatabase) only has data through 2023.
TRAINING_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
EXCLUDED_YEARS = [2020, 2024]
LEAGUES = ["AL", "NL"]

# Eligibility (Phase 1: fixed threshold)
TRAINING_MIN_IP = 50

# Vote share denominator: 30 voters × 7 points for first-place
MAX_BBWAA_POINTS = 210

# Feature columns (26 total: 10 traditional + 6 sabermetric + 6 Statcast + 4 context)
TRADITIONAL_COLS = ["W", "L", "ERA", "IP", "K", "BB", "WHIP", "CG", "ShO", "SV"]
SABERMETRIC_COLS = ["fWAR", "FIP", "xFIP", "K-BB%", "ERA-", "FIP-"]
STATCAST_COLS = ["xERA", "xwOBA_against", "Stuff+", "Location+", "Barrel%", "HardHit%"]
CONTEXT_COLS = ["role_SP", "league_AL", "team_winning_pct", "RS_per_9"]
FEATURE_COLS = TRADITIONAL_COLS + SABERMETRIC_COLS + STATCAST_COLS + CONTEXT_COLS
assert len(FEATURE_COLS) == 26  # 10 traditional + 6 sabermetric + 6 statcast + 4 context

# Paths
AWARDS_PARQUET = HISTORICAL_DIR / "awards_history.parquet"
TRAINING_PARQUET = HISTORICAL_DIR / "training_2015_2023.parquet"
GBR_MODEL_PATH = MODELS_DIR / "voter_model_gbr_v1.pkl"
RIDGE_MODEL_PATH = MODELS_DIR / "voter_model_ridge_v1.pkl"
CALIBRATOR_PATH = MODELS_DIR / "calibrator_v1.pkl"
BACKTEST_REPORT_PATH = REPORTS_DIR / "backtest_v1.md"

# KPI gates (per spec section 1.3)
# Denominator adjusted for 8 years × 2 leagues = 16 winner slots (was 18 for 9 years).
KPI_TARGETS = {
    "winner_hits_min": 12,        # ~75% of 16 winner slots
    "winner_hits_total": 16,      # 8 years × 2 leagues
    "podium_overlap_avg_min": 2.0,
    "podium_overlap_avg_max": 3.0,
    "top10_overlap_avg_min": 7.0,
    "top10_overlap_avg_max": 10.0,
}
