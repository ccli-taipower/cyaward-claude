# Backtest Report — GradientBoostingRegressor v1

**Overall verdict:** PASS

## KPI Summary (LOOCV)

| Tier | Metric | Target | Result | Status |
|---|---|---|---|---|
| Tier 1 | Winner hits | >= 12 / 16 | 13 / 16 | PASS |
| Tier 2 | Podium overlap avg | >= 1.9 / 3 | 1.94 / 3 | PASS |
| Tier 3 | Top-10 overlap avg | >= 7.0 / 10 | 8.12 / 10 | PASS |

**Vote-share MAE (LOOCV):** 0.0085

## Outlier Cases (predicted top-1 != actual winner)

| Year | League | Predicted | Actual |
|---|---|---|---|
| 2018 | AL | Justin Verlander | Blake Snell |
| 2021 | AL | Carlos Rodón | Robbie Ray |
| 2021 | NL | Walker Buehler | Corbin Burnes |

## Time-Series Split Sanity (train 2015-2022 / val 2023 / test 2024)

Time-split predictions count: 357
(See `models/voter_model_*_v1.pkl` for the trained models.)

## Ridge Baseline (LOOCV)

Winner hits: **7 / 16** (GBR: 13)
