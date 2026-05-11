# Backtest Report — GradientBoostingRegressor v1

**Overall verdict:** PASS

## KPI Summary (LOOCV)

| Tier | Metric | Target | Result | Status |
|---|---|---|---|---|
| Tier 1 | Winner hits | >= 15 / 20 | 15 / 20 | PASS |
| Tier 2 | Podium overlap avg | >= 1.9 / 3 | 1.95 / 3 | PASS |
| Tier 3 | Top-10 overlap avg | >= 7.0 / 10 | 8.20 / 10 | PASS |

**Vote-share MAE (LOOCV):** 0.0076

## Outlier Cases (predicted top-1 != actual winner)

| Year | League | Predicted | Actual |
|---|---|---|---|
| 2016 | NL | Clayton Kershaw | Max Scherzer |
| 2018 | AL | Justin Verlander | Blake Snell |
| 2019 | AL | Gerrit Cole | Justin Verlander |
| 2021 | AL | Carlos Rodón | Robbie Ray |
| 2021 | NL | Walker Buehler | Corbin Burnes |

## Time-Series Split Sanity (train 2015-2022 / val 2023 / test 2024)

Time-split predictions count: 357
(See `models/voter_model_*_v1.pkl` for the trained models.)

## Ridge Baseline (LOOCV)

Winner hits: **10 / 20** (GBR: 15)
