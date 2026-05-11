# Backtest Report — GradientBoostingRegressor v1

**Overall verdict:** FAIL

## KPI Summary (LOOCV)

| Tier | Metric | Target | Result | Status |
|---|---|---|---|---|
| Tier 1 | Winner hits | >= 12 / 16 | 8 / 16 | FAIL |
| Tier 2 | Podium overlap avg | >= 2.0 / 3 | 2.00 / 3 | PASS |
| Tier 3 | Top-10 overlap avg | >= 7.0 / 10 | 7.88 / 10 | PASS |

**Vote-share MAE (LOOCV):** 0.0098

## Outlier Cases (predicted top-1 != actual winner)

| Year | League | Predicted | Actual |
|---|---|---|---|
| 2016 | NL | Clayton Kershaw | Max Scherzer |
| 2017 | AL | Chris Sale | Corey Kluber |
| 2018 | AL | Justin Verlander | Blake Snell |
| 2018 | NL | Max Scherzer | Jacob deGrom |
| 2019 | AL | Gerrit Cole | Justin Verlander |
| 2021 | AL | Carlos Rodón | Robbie Ray |
| 2021 | NL | Walker Buehler | Corbin Burnes |
| 2022 | AL | Shohei Ohtani | Justin Verlander |

## Time-Series Split Sanity (train 2015-2022 / val 2023 / test 2024)

Time-split predictions count: 357
(See `models/voter_model_*_v1.pkl` for the trained models.)

## Ridge Baseline (LOOCV)

Winner hits: **8 / 16** (GBR: 8)
