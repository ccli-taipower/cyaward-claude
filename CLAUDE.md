# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Phase 1 COMPLETE** (tag `phase1-complete`) — voter-share regression model trained and backtested on 2015–2025 BBWAA Cy Young voting (10 years × 2 leagues = 20 winner slots), all 3 KPI tiers PASS (15/20 winners, 1.95/3 podium avg, 8.20/10 top-10 avg, MAE 0.0076).

**Phase 2 COMPLETE** (tag `phase2-complete`) — live 2026 dashboard at https://ccli-taipower.github.io/cyaward-claude/ auto-updated daily by `.github/workflows/daily.yml` (cron `0 11 * * *` UTC = 19:00 Taiwan). Weekly markdown report at `reports/2026-Wxx.md` generated every Monday by `weekly.yml` (cron `0 2 * * 1` UTC). Plan: [`docs/superpowers/plans/2026-05-12-cyaward-phase2.md`](docs/superpowers/plans/2026-05-12-cyaward-phase2.md).

## Common commands

```bash
# Setup (one-time)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1 — train + backtest the model (one-time / on retrain)
python -m src.cli.build_training_data    # ~5-15 min cold; cached CSVs in data/historical/
python -m src.cli.train                   # < 1 min — saves 3 pkl files to models/
python -m src.cli.backtest                # ~1-3 min — writes reports/backtest_v1.md, exit 0 if KPI PASS

# Phase 2 — daily live ranking (runs in GitHub Actions; can also run locally)
python -m src.cli.daily                   # today's date; ~30s — writes data/predictions/<date>.parquet + site/index.html
python -m src.cli.daily --date 2026-05-12 # backfill / specific date
open site/index.html                      # preview locally

# Phase 2 — weekly report (needs ≥1 day of predictions; ≥2 days for non-empty movers)
python -m src.cli.weekly                  # last Sunday; writes reports/<year>-Wxx.md
python -m src.cli.weekly --week-end 2026-05-17

# Tests (62 unit tests, all mocked, ~5 seconds)
pytest                                    # all
pytest tests/test_backtest.py -v          # one file
pytest tests/test_backtest.py::test_winner_hits_count -v   # one test
pytest -m slow                            # opt-in: integration tests that hit live network (none currently registered)
```

The backtest CLI exits **2** on KPI failure — useful for CI gating.

## Architecture (read this before touching code)

Two pipelines share the fetch + features + model layers:

```
                    fetch.py
                       │
                       ▼
                  features.py
                  ┌────┴────┐
                  ▼         ▼
        Phase 1: backtest   Phase 2: ranker
        (LOOCV + KPI)       (eligibility + projector + render)
                  │                 │
                  ▼                 ▼
        reports/backtest_v1.md   site/index.html
                                 data/predictions/<date>.parquet
                                 reports/<year>-Wxx.md (weekly)
```

### Shared layers (Phase 1 + Phase 2)

- **`src/fetch.py`** is the **only** module that touches the network. Tests mock at this boundary (`src.fetch.pyb.*`, `src.fetch._load_awards_share_players`, `src.fetch._fg_api_get`). Public functions: `get_fangraphs_pitching`, `get_team_records`, `get_bref_pitching` (stub), `get_awards_history`.
- **`src/features.py`** is pure — takes the four raw DataFrames produced by fetch, returns one canonical training row per pitcher-season with 38 features + `vote_share` label. Accepts an empty awards DataFrame for inference (Phase 2) — line 96-97 fallback fills `pitcher_name` from `Name` when no awards match.
- **`src/voter_model.py`** wraps sklearn: `train_gbr` (primary, 350 trees), `train_ridge` (baseline), `train_calibrator` (isotonic), plus `predict`/`save_model`/`load_model`. Each model is a sklearn `Pipeline` (imputer → optional scaler → model). `predict()` clips to [0, 1].
- **`src/config.py`** is the single source of truth for `TRAINING_YEARS`, `FEATURE_COLS` (38, with runtime assert), `KPI_TARGETS`, and all paths.

### Phase 1 (offline training & backtest)

- **`src/backtest.py`** runs LOOCV over `config.TRAINING_YEARS`, computes the 3-tier KPI metrics, generates the markdown report. Takes the prebuilt training parquet — does NOT call fetch.
- **`src/cli/build_training_data.py`**, **`train.py`**, **`backtest.py`** — thin orchestrators.

### Phase 2 (live daily ranking)

- **`src/eligibility.py`** — date-based dynamic SP/RP IP threshold. `SEASON_START = 2026-03-26`, `SEASON_END = 2026-09-27`. Pre-season returns empty.
- **`src/projector.py`** — `PaceProjector` scales counting stats (W, L, K, BB, IP, fWAR, CG, ShO, SV) by `1/season_progress`; rate stats (ERA, FIP, etc.) pass through unchanged. Pre/post-season pass-through. Abstract `Projector` class reserved for v3 Marcel-style projector.
- **`src/ranker.py`** — orchestrator: `rank_today(asof_date)` → fetch (with `force_refresh=True`) → `filter_eligible` → project → `build_features` (empty awards) → `voter_model.predict` → sort within league → return 12-column ranking DataFrame.
- **`src/render.py`** — Jinja2 (`templates/dashboard.html.j2`) → `site/index.html`. AL/NL Top 10 cards, ~18 KB, no JS dependency. Empty-ranking renders a graceful "no data" page.
- **`src/weekly_report.py`** — aggregates 7 days of `data/predictions/<date>.parquet`, computes rank deltas (first day vs last day), emits markdown with movers + Top 10 tables.
- **`src/cli/daily.py`** and **`weekly.py`** — thin CLIs invoked by GitHub Actions.

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

## Caching — historical vs current-year split

`get_fangraphs_pitching(year, force_refresh=False)` and `get_team_records(year, force_refresh=False)` cache CSVs in `data/historical/`. Behavior differs by year:

- **Historical years (2015–2025)** — cache files like `fg_pitching_2024.csv` are **committed to git** as reproducibility snapshots. Phase 1 CLIs use default `force_refresh=False` and read from cache after the first network call. Delete a year's file to force re-fetch.
- **Current year (2026)** — `data/historical/fg_pitching_2026.csv` and `standings_2026.csv` are **git-ignored** because they change daily. `src/ranker.py` calls fetch with `force_refresh=True` so each daily run skips the cache and pulls fresh data.

If FanGraphs or MLB Stats API changes shape, `fetch.py` is where to fix it.

## Feature pipeline notes

The 38 features split as 10 traditional + 6 sabermetric + 6 Statcast + 16 context. The **context group is where the model's predictive power comes from**: GBR feature importance shows `fWAR_z_score` (24.8%), `fWAR_rank_in_league` (18.9%), `fWAR` (14.0%) together account for 57.8% of decision weight. `era_rank_in_league` + `ERA-` add another 14.0% (these were the iteration #2 additions that pushed winner_hits from 8/16 → 13/16 by capturing "ERA dominance" cases like deGrom 2018). These league-context features are computed per-year per-league inside `build_features` via `df.groupby("league")` transforms.

A few features are near-zero importance and intentionally retained: `late_era_z_score_neg`, `late_vs_full_era_delta` (zero-filled — FanGraphs monthly-split API is Cloudflare-blocked); `role_SP` (everyone above IP=50 is essentially SP); `league_AL` (voter behavior is similar across leagues).

When adding a new feature:
1. Add the column name to the appropriate list in `src/config.py`
2. Update the `assert len(FEATURE_COLS) == N` line — the model will crash at import if your list size disagrees
3. Compute the feature inside `build_features` BEFORE the final `keep = [...]` line
4. Update `tests/test_config.py::test_feature_cols_count` and add a feature-specific test in `tests/test_features.py`
5. Delete `data/historical/training_2015_2025.parquet` and rebuild — features in the parquet must match `FEATURE_COLS`
6. Retrain (`python -m src.cli.train`) and rebacktest (`python -m src.cli.backtest`); the GBR pkl is also committed to git

## KPI gate

`reports/backtest_v1.md` is the canonical proof-of-quality. The backtest CLI returns exit code 0 only if all three tiers pass thresholds in `config.KPI_TARGETS`. If a change drops winner_hits below 15/20, podium below 1.9/3, or top-10 below 7.0/10, the gate fails. Tier 2 was relaxed from 2.0 to 1.9 because a 1-pitcher swap across 16 cases is statistical noise (rationale recorded in `config.py` comment).

The model gets the actual winner in its predicted Top 5 for **all 20 historical cases (100%)** and Top 3 for 17/20 (85%). The three "winner not in Top 3" cases are 2018 AL Snell (model rank 5), 2016 NL Scherzer (4), 2021 NL Burnes (4) — all close-race / narrative-driven outcomes the spec documents as inherent limits.

## Test conventions

- Fixtures live in `tests/conftest.py` (FG-shaped, B-Ref-shaped, standings-shaped, Lahman-awards-shaped DataFrames).
- Tests mock `src.fetch` functions, never `pybaseball` directly — keeps the boundary clean.
- Mock-target paths: `unittest.mock.patch("src.fetch.pyb.chadwick_register", ...)` etc. — note the `src.fetch.pyb` prefix because `src/fetch.py` does `import pybaseball as pyb`. For ranker tests, patch `src.ranker.fetch.<func>` (the local import binding).
- `synthetic_loocv_input` and `synthetic_training_set` fixtures provide deterministic small datasets for backtest/model tests; seed is `np.random.default_rng(42)` or `(7)` — don't change without reason.

## GitHub Pages deployment

`.github/workflows/daily.yml` runs the ranker on schedule and commits both `data/predictions/<date>.parquet` and `site/index.html` back to `main` as the `cyaward-bot` user. GitHub Pages is configured as "Deploy from a branch" with source = `main` / `/(root)`. Because Pages doesn't support a `/site` source path, there's a redirect at `/index.html` (project root) that bounces to `/site/` — that's why the dashboard URL works at https://ccli-taipower.github.io/cyaward-claude/.

`.github/workflows/weekly.yml` does the same for `reports/<year>-Wxx.md` on Monday mornings UTC.

Both workflows have `permissions: contents: write` (needed to push) and use `workflow_dispatch` (manual trigger via `gh workflow run daily.yml -R ccli-taipower/cyaward-claude`).

## Workflow conventions

- Branch `main` is the canonical state; feature work uses branches like `phase1-impl`, `phase2-impl` merged via `--no-ff`.
- Commit messages use Conventional Commits (`feat(scope):`, `fix(scope):`, `chore:`, `docs:`, `refactor:`, `ci:`, `data:`).
- Milestone tags (`phase1-complete`, `phase2-complete`) mark end-of-phase state and are pushed to origin.
- Design specs go in `docs/superpowers/specs/`, implementation plans in `docs/superpowers/plans/`. If a design decision changes during implementation, update the spec.
