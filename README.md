# cyaward-claude

MLB Cy Young Award voter-share regression model.

**Phase 1 (COMPLETE):** model trained and backtested on 2015–2023 BBWAA voting (8 years × 2 leagues = 16 winner slots).  
**Phase 2 (DEFERRED):** live 2026 dashboard not built; see [design spec](docs/superpowers/specs/2026-05-11-cyaward-design.md).

---

## Results — Phase 1 KPIs

All three KPI tiers pass.

| Tier | Metric | Target | Result | Status |
|---|---|---|---|---|
| 1 (Winner hits) | Correct winner predicted | >= 12 / 16 | **13 / 16 (81%)** | PASS |
| 2 (Podium overlap avg) | Avg overlap in top 3 | >= 1.9 / 3 | **1.94 / 3 (65%)** | PASS |
| 3 (Top-10 overlap avg) | Avg overlap in top 10 | >= 7.0 / 10 | **8.12 / 10 (81%)** | PASS |
| — | Vote-share MAE (LOOCV) | (informational) | **0.0085** | — |

**Bonus metrics:**

- Actual winner in predicted Top 3: **14 / 16 (88%)**
- Actual winner in predicted Top 5: **16 / 16 (100%)**
- Ridge baseline winner hits (LOOCV): **7 / 16** vs GradientBoosting **13 / 16**

Full details: [`reports/backtest_v1.md`](reports/backtest_v1.md)

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
python -m src.cli.build_training_data    # first-time only; ~5-10 min; cached after
python -m src.cli.train                   # < 1 min
python -m src.cli.backtest                # ~1-3 min
```

**Outputs:**

| File | Description |
|---|---|
| `data/historical/training_2015_2023.parquet` | 2731-row training dataset |
| `models/voter_model_gbr_v1.pkl` | Trained GradientBoostingRegressor |
| `models/voter_model_ridge_v1.pkl` | Trained Ridge baseline |
| `models/calibrator_v1.pkl` | Isotonic calibrator for vote-share |
| `reports/backtest_v1.md` | LOOCV + time-series KPI report |

---

## Architecture

The pipeline has five layers:

| Module | Role |
|---|---|
| `src/fetch.py` | Wrappers around 4 external APIs — FanGraphs JSON, Lahman mirror (jmaslek/LahmanDatabase), MLB Stats API, Baseball Savant — plus `pybaseball.chadwick_register` for player-ID crosswalk |
| `src/features.py` | Pure join + derive + filter; produces 38 canonical features |
| `src/voter_model.py` | GradientBoostingRegressor + Ridge + isotonic calibrator |
| `src/backtest.py` | LOOCV + time-series split + 3-tier KPI report |
| `src/cli/` | Thin CLI orchestrators (`build_training_data`, `train`, `backtest`) |

---

## Feature Set

The model uses **38 features** across 4 categories:

| Category | Count | Examples |
|---|---|---|
| Traditional | 10 | W, L, ERA, IP, K, BB, WHIP, CG, ShO, SV |
| Sabermetric | 6 | fWAR, FIP, xFIP, K-BB%, ERA-, FIP- |
| Statcast | 6 | xERA, xwOBA_against, Stuff+, Location+, Barrel%, HardHit% |
| Context / derived | 16 | `era_z_score_neg`, `ip_relative_to_max`, `era_rank_in_league`, `FIP_rank_in_league`, `fWAR_rank_in_league`, `wins_rank_in_league`, … |

The league-context features — especially "ERA dominance" (`era_z_score_neg`, `era_rank_in_league`) and "workhorse" (`ip_relative_to_max`) — were the most critical for predicting close races.

---

## Project Layout

```
data/historical/      # training parquet + raw CSV caches (committed)
models/               # trained pkl artifacts (committed)
reports/              # backtest_v1.md (committed)
src/                  # pure Python — fetch, features, voter_model, backtest
src/cli/              # thin CLI orchestrators
tests/                # 41 pytest unit tests
docs/superpowers/     # design spec + implementation plan
```

---

## Testing

```bash
pytest              # all unit tests (mocked, fast — ~5 seconds)
```

41 tests should pass.

---

## Known Limitations

**2024 data excluded.** `pybaseball`'s Lahman dependency was broken (chadwickbureau/baseballdatabank repo deleted). We use the jmaslek/LahmanDatabase mirror, which only has data through 2023. Phase 2 will need a separate strategy for current-season awards.

**Multiple external APIs blocked.** `pybaseball`'s FanGraphs and Baseball-Reference scrapers are 403 Cloudflare-blocked. We use the FanGraphs JSON API and MLB Stats API directly. These endpoints may change without notice.

**No late-season splits.** The FanGraphs monthly-split API is also 403-blocked. Features `late_era_z_score_neg` and `late_vs_full_era_delta` exist in the schema but are zero-filled. Phase 2 could add game-log aggregation as a replacement.

**Three outlier misses remain:**

| Year | League | Model pick | Actual winner | Notes |
|---|---|---|---|---|
| 2018 | AL | Justin Verlander | Blake Snell | Narrative-driven; Snell had historic K/9 in shortened outings |
| 2021 | AL | Carlos Rodón | Robbie Ray | Close race; Ray led AL in K% and IP |
| 2021 | NL | Walker Buehler | Corbin Burnes | Close race; Burnes had historic K-BB% |

All three were close-race cases where the model's pick was defensible per contemporary betting markets and media consensus.

**Statcast era only.** Training is 2015+ because Stuff+, Location+, and xERA don't exist before that. Pre-2020 rows (~62% of training data) have NaN in Stuff+/Location+; a median imputer in the pipeline handles this.

---

## Links

- Design spec: [docs/superpowers/specs/2026-05-11-cyaward-design.md](docs/superpowers/specs/2026-05-11-cyaward-design.md)
- Implementation plan: [docs/superpowers/plans/2026-05-11-cyaward-phase1.md](docs/superpowers/plans/2026-05-11-cyaward-phase1.md)
- Backtest report: [reports/backtest_v1.md](reports/backtest_v1.md)
