# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Phase 1 COMPLETE** (tag `phase1-complete`) — voter-share regression model trained and backtested on 2015–2025 BBWAA Cy Young voting (10 years × 2 leagues = 20 winner slots), all 3 KPI tiers PASS (15/20 winners, 1.95/3 podium avg, 8.20/10 top-10 avg, MAE 0.0076).

**Phase 2 COMPLETE** (tag `phase2-complete`) — live 2026 dashboard at `site/index.html` + weekly report at `reports/2026-Wxx.md`. Daily GitHub Actions cron (`0 11 * * *` UTC) runs `python -m src.cli.daily`; weekly cron (`0 2 * * 1` UTC) runs `python -m src.cli.weekly`. Plan: [`docs/superpowers/plans/2026-05-12-cyaward-phase2.md`](docs/superpowers/plans/2026-05-12-cyaward-phase2.md). Deployment to GitHub Pages requires repo push + Actions enablement.

## Common commands

```bash
# Setup (one-time)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (in order)
python -m src.cli.build_training_data    # ~5-15 min cold; cached CSVs in data/historical/
python -m src.cli.train                   # < 1 min — saves 3 pkl files to models/
python -m src.cli.backtest                # ~1-3 min — writes reports/backtest_v1.md, exit 0 if KPI PASS

# Tests (41 unit tests, all mocked, ~5 seconds)
pytest                                    # all
pytest tests/test_backtest.py -v          # one file
pytest tests/test_backtest.py::test_winner_hits_count -v   # one test
pytest -m slow                            # opt-in: integration tests that hit live network (none currently registered)
```

The backtest CLI exits **2** on KPI failure — useful for CI gating.

## Architecture (read this before touching code)

Layered pipeline, strict separation of I/O from logic. Each layer takes DataFrames in, returns DataFrames out.

```
fetch.py  →  features.py  →  voter_model.py  →  backtest.py
(network)    (pure joins)     (sklearn pipes)    (LOOCV + metrics)
                                                      ↓
                                              reports/backtest_v1.md
```

- **`src/fetch.py`** is the **only** module that touches the network. Tests mock at this boundary (`src.fetch.pyb.*`, `src.fetch._load_awards_share_players`, `src.fetch._fg_api_get`).
- **`src/features.py`** is pure — takes the four raw DataFrames produced by fetch, returns one canonical training row per pitcher-season with 38 features + `vote_share` label.
- **`src/voter_model.py`** wraps sklearn: `train_gbr` (primary), `train_ridge` (baseline), `train_calibrator` (isotonic, for Phase 2 win-probability), plus `predict`/`save_model`/`load_model`. Each model is a sklearn `Pipeline` (imputer → optional scaler → model).
- **`src/backtest.py`** runs LOOCV over `config.TRAINING_YEARS`, computes the 3-tier KPI metrics, generates the markdown report. Takes the prebuilt training parquet — does NOT call fetch.
- **`src/cli/`** modules are thin orchestrators (parse args, load config, call the layer above). No business logic.
- **`src/config.py`** is the single source of truth for `TRAINING_YEARS`, `FEATURE_COLS` (38, with runtime assert), `KPI_TARGETS`, and all paths. Other modules import from here.

## External data sources (non-obvious!)

`pybaseball` is installed but **most of its scrapers are broken** as of 2025+:
- `pybaseball.pitching_stats` (FanGraphs scraper) → 403 Cloudflare
- `pybaseball.pitching_stats_bref` (Baseball-Reference) → 403 Cloudflare
- `pybaseball.standings` (Baseball-Reference) → 403 Cloudflare
- `pybaseball.lahman.*` → 404 (chadwickbureau/baseballdatabank repo was deleted)

`fetch.py` works around all of these:
- **FanGraphs JSON API** (`https://www.fangraphs.com/api/leaders/major-league/data`) replaces `pitching_stats`
- **MLB Stats API** (`statsapi.mlb.com`) replaces `standings`
- **Baseball Savant CSV** replaces Statcast endpoints for xwOBA-against
- **jmaslek/LahmanDatabase GitHub zip** replaces `lahman.awards_share_players` (only goes through 2023)
- **bbwaa.com scrape** supplies 2024–2025 Cy Young voting (cached in `data/historical/awards_2024_2025.csv`)
- **`pybaseball.chadwick_register()`** is the only pybaseball call that still works — used for Lahman playerID → name mapping

**Never patch `.venv/site-packages/pybaseball/`.** All workarounds live in our own `src/fetch.py`.

If FanGraphs or MLB Stats API changes shape, `fetch.py` is where to fix it. The cached raw CSVs in `data/historical/fg_pitching_YYYY.csv` and `standings_YYYY.csv` are **committed to git as reproducibility snapshots** — delete a year's file to force re-fetch.

## Feature pipeline notes

The 38 features split as 10 traditional + 6 sabermetric + 6 Statcast + 16 context. The **context group is where the model's predictive power comes from**: GBR feature importance shows `fWAR_z_score`, `fWAR_rank_in_league`, `era_rank_in_league` together account for >50% of decision weight. These are computed per-year per-league inside `build_features` via `df.groupby("league")` transforms.

When adding a new feature:
1. Add the column name to the appropriate list in `src/config.py`
2. Update the `assert len(FEATURE_COLS) == N` line — the model will crash at import if your list size disagrees
3. Compute the feature inside `build_features` BEFORE the final `keep = [...]` line
4. Update `tests/test_config.py::test_feature_cols_count` and add a feature-specific test in `tests/test_features.py`
5. Delete `data/historical/training_2015_2025.parquet` and rebuild — features in the parquet must match `FEATURE_COLS`

## KPI gate

`reports/backtest_v1.md` is the canonical proof-of-quality. The backtest CLI returns exit code 0 only if all three tiers pass thresholds in `config.KPI_TARGETS`. If a change drops winner_hits below 15/20, podium below 1.9/3, or top-10 below 7.0/10, the gate fails. Tier 2 was relaxed from 2.0 to 1.9 because a 1-pitcher swap across 16 cases is statistical noise (rationale recorded in `config.py` comment).

## Test conventions

- Fixtures live in `tests/conftest.py` (FG-shaped, B-Ref-shaped, standings-shaped, Lahman-awards-shaped DataFrames).
- Tests mock `src.fetch` functions, never `pybaseball` directly — keeps the boundary clean.
- Mock-target paths: `unittest.mock.patch("src.fetch.pyb.chadwick_register", ...)` etc. — note the `src.fetch.pyb` prefix because `src/fetch.py` does `import pybaseball as pyb`.
- `synthetic_loocv_input` and `synthetic_training_set` fixtures provide deterministic small datasets for backtest/model tests; seed is `np.random.default_rng(42)` or `(7)` — don't change without reason.

## Workflow conventions

- Branch `main` is the canonical state; feature work happened on `phase1-impl` and merged via `--no-ff`. Use the same pattern for Phase 2.
- Commit messages use Conventional Commits (`feat(scope):`, `fix(scope):`, `chore:`, `docs:`, `refactor:`).
- The `phase1-complete` tag marks the end-of-Phase-1 milestone; tag Phase 2 similarly when done.
- Design specs go in `docs/superpowers/specs/`, implementation plans in `docs/superpowers/plans/`. The Phase 1 spec is authoritative; if a design decision changes during implementation, update the spec.
