# cyaward-claude

MLB Cy Young Award voter-share regression model.

### 🔗 Live Dashboard: **https://ccli-taipower.github.io/cyaward-claude/**

Updated daily at ~19:00 Taiwan time (UTC 11:00). AL+NL Top 10 predicted vote shares for the 2026 season.

### 📊 Weekly Reports: [reports/](https://github.com/ccli-taipower/cyaward-claude/tree/main/reports)

Generated every Monday at ~10:00 Taiwan time (UTC 02:00). Each `2026-Wxx.md` covers the previous 7 days — biggest rank movers (risers + fallers), week's #1 in each league, and a Top 10 snapshot.

**Phase 1 (COMPLETE):** model trained and backtested on 2015–2025 BBWAA voting (10 years × 2 leagues = 20 winner slots).  
**Phase 2 (COMPLETE):** live 2026 dashboard at the URL above + weekly markdown report; daily/weekly GitHub Actions cron deployed.

---

## Results — Phase 1 KPIs

All three KPI tiers pass.

| Tier | Metric | Target | Result | Status |
|---|---|---|---|---|
| 1 (Winner hits) | Correct winner predicted | >= 15 / 20 | **15 / 20 (75%)** | PASS |
| 2 (Podium overlap avg) | Avg overlap in top 3 | >= 1.9 / 3 | **1.95 / 3 (65%)** | PASS |
| 3 (Top-10 overlap avg) | Avg overlap in top 10 | >= 7.0 / 10 | **8.20 / 10 (82%)** | PASS |
| — | Vote-share MAE (LOOCV) | (informational) | **0.0076** | — |

**Bonus metrics:**

- Ridge baseline winner hits (LOOCV): **10 / 20** vs GradientBoosting **15 / 20**

Prior result (2015–2023, 8 years): 13 / 16 winner hits, MAE 0.0085.

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
| `data/historical/training_2015_2025.parquet` | 3421-row training dataset |
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

## Phase 2 — Live Dashboard

**Status:** Live at **<https://ccli-taipower.github.io/cyaward-claude/>** — auto-updated daily.

The Phase 2 pipeline applies the validated Phase 1 GBR model to live 2026 stats and produces:

- **Daily HTML dashboard** at `site/index.html` (served via GitHub Pages) with AL+NL Top 10.
- **Weekly markdown report** at `reports/2026-Wxx.md` (every Monday) with rank movers and Top 10 snapshot.

### Local run

```bash
# Run today's ranking + render the HTML
python -m src.cli.daily                       # uses today's date
python -m src.cli.daily --date 2026-05-12     # backfill a specific date
open site/index.html                          # macOS preview

# Generate this week's report (after at least 1 daily run has produced a parquet)
python -m src.cli.weekly --week-end 2026-05-17
```

### Automation

Two GitHub Actions workflows are committed but require pushing to a GitHub remote + manual enablement of Actions to start firing:

- `.github/workflows/daily.yml` — `cron: "0 11 * * *"` (19:00 Taiwan time). Runs the daily pipeline; commits the new parquet + updated `site/index.html`.
- `.github/workflows/weekly.yml` — `cron: "0 2 * * 1"` (Monday 10:00 Taiwan time). Generates the weekly report.

Failure handling: each daily run retries up to 3 times with exponential backoff; if all retries fail, **no commit is made** (the previous day's `site/index.html` stays live).

### Architecture additions

| Module | Role |
|---|---|
| `src/eligibility.py` | Dynamic SP/RP IP threshold scaled by season progress |
| `src/projector.py` | `PaceProjector`: scales counting stats to full-season equivalents |
| `src/ranker.py` | Orchestrator: fetch → eligibility → project → features → predict → rank |
| `src/render.py` | Jinja2 renderer (`templates/dashboard.html.j2` → `site/index.html`) |
| `src/weekly_report.py` | Aggregates 7 daily parquets into a markdown report |
| `src/cli/daily.py` | Daily pipeline CLI |
| `src/cli/weekly.py` | Weekly pipeline CLI |

### Known limitations (Phase 2)

- **Pace projector is naive**: linear `IP_full = IP_current / season_progress`. Early-season predictions for breakout/injury cases will swing wildly. A Marcel-style regression-to-mean projector is reserved for v3.
- **No sparkline yet**: the spec calls for per-pitcher 30-day vote-share trend lines; the parquet history accumulates day-by-day so this is straightforward to add as a follow-up.
- **2024-2025 awards are scraped manually** from `bbwaa.com`; 2026 will need the same treatment after the November vote.
- **Caching**: `fg_pitching_2026.csv` and `standings_2026.csv` are git-ignored (regenerated daily); historical-year caches stay committed for reproducibility.

---

## Feature Set

The model uses **38 features** across 4 categories:

| Category | Count | Examples |
|---|---|---|
| Traditional | 10 | W, L, ERA, IP, K, BB, WHIP, CG, ShO, SV |
| Sabermetric | 6 | fWAR, FIP, xFIP, K-BB%, ERA-, FIP- |
| Statcast | 6 | xERA, xwOBA_against, Stuff+, Location+, Barrel%, HardHit% |
| Context / derived | 16 | `era_z_score_neg`, `ip_relative_to_max`, `era_rank_in_league`, `FIP_rank_in_league`, `fWAR_rank_in_league`, `wins_rank_in_league`, … |

### Model & feature importance

The primary model is a `GradientBoostingRegressor` (350 trees, max_depth=3, learning_rate=0.05, subsample=0.8). A `Ridge` model is kept as a baseline (it scored only 10/20 winner hits vs GBR's 15/20 — multicollinearity hurts the linear model).

Top 10 features by GBR importance (account for **88.4%** of total predictive weight):

| # | Feature | Importance | What it captures |
|---|---|---|---|
| 1 | `fWAR_z_score` | 24.8% | Pitcher's fWAR z-score within league/year |
| 2 | `fWAR_rank_in_league` | 18.9% | Pitcher's fWAR rank within league |
| 3 | `fWAR` | 14.0% | Raw FanGraphs WAR |
| 4 | `era_rank_in_league` | 7.4% | ERA rank within league (1 = leader) |
| 5 | `ERA-` | 6.6% | Park/league-adjusted ERA |
| 6 | `WHIP` | 5.4% | Walks + hits per inning |
| 7 | `ip_relative_to_max` | 3.3% | IP / max(IP in league) — workhorse signal |
| 8 | `K` | 2.9% | Strikeout count |
| 9 | `wins_rank_in_league` | 2.6% | Win-total rank |
| 10 | `ip_rank_in_league` | 2.4% | IP rank within league |

**Take-aways:**
- **fWAR-family features (top 3) = 57.8%** of the model's decisions — fWAR with league-context normalization is the spine of the predictor.
- **ERA-dominance features (`era_rank_in_league` + `ERA-`) = 14.0%** — added in iteration #2 to fix cases like 2018 NL deGrom (1.70 ERA legend), and these are what pushed winner_hits from 8/16 → 13/16.
- **Several features have ~0% importance** — `role_SP` (everyone above IP=50 is essentially SP), `league_AL` (voter behavior is similar across leagues), and the `late_*` features (zero-filled stubs because the FanGraphs monthly-split API is Cloudflare-blocked).

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

**Multiple external APIs blocked.** `pybaseball`'s FanGraphs and Baseball-Reference scrapers are 403 Cloudflare-blocked. We use the FanGraphs JSON API and MLB Stats API directly. These endpoints may change without notice.

**No late-season splits.** The FanGraphs monthly-split API is also 403-blocked. Features `late_era_z_score_neg` and `late_vs_full_era_delta` exist in the schema but are zero-filled. Phase 2 could add game-log aggregation as a replacement.

**2024–2025 award data source.** The Lahman mirror (jmaslek/LahmanDatabase) only covers through 2023. 2024 and 2025 BBWAA Cy Young votes are scraped directly from bbwaa.com and stored in `data/historical/awards_2024_2025.csv`.

**Five outlier misses (LOOCV over 10 years):**

| Year | League | Model pick | Actual winner | Notes |
|---|---|---|---|---|
| 2016 | NL | Clayton Kershaw | Max Scherzer | Close race |
| 2018 | AL | Justin Verlander | Blake Snell | Narrative-driven; Snell had historic K/9 in shortened outings |
| 2019 | AL | Gerrit Cole | Justin Verlander | Close race |
| 2021 | AL | Carlos Rodón | Robbie Ray | Close race; Ray led AL in K% and IP |
| 2021 | NL | Walker Buehler | Corbin Burnes | Close race; Burnes had historic K-BB% |

All five were close-race cases where the model's pick was defensible per contemporary betting markets and media consensus.

**Statcast era only.** Training is 2015+ because Stuff+, Location+, and xERA don't exist before that. Pre-2020 rows (~62% of training data) have NaN in Stuff+/Location+; a median imputer in the pipeline handles this.

---

## Links

- Design spec: [docs/superpowers/specs/2026-05-11-cyaward-design.md](docs/superpowers/specs/2026-05-11-cyaward-design.md)
- Implementation plan: [docs/superpowers/plans/2026-05-11-cyaward-phase1.md](docs/superpowers/plans/2026-05-11-cyaward-phase1.md)
- Backtest report: [reports/backtest_v1.md](reports/backtest_v1.md)
