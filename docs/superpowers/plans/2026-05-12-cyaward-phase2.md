# cyaward-claude Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the validated Phase 1 GBR model to live 2026 MLB pitcher data, producing a daily-updated GitHub Pages dashboard (AL+NL Top 10 with predicted vote shares) and a weekly markdown report.

**Architecture:** Reuse Phase 1 fetch wrappers (FanGraphs JSON API, MLB Stats API, Baseball Savant) parameterized to year=2026. Add Stage A pace projector that scales current cumulative stats to full-season equivalents. Build features the same way as training, predict with the Phase 1 GBR pkl, render a static HTML dashboard via Jinja2, and emit a weekly markdown report. Two GitHub Actions cron jobs publish to GitHub Pages.

**Tech Stack:** Python 3.10+, existing Phase 1 dependencies + Jinja2 for templating. No new external services. Local-first MVP, then GitHub Pages.

**Spec:** [docs/superpowers/specs/2026-05-11-cyaward-design.md](../specs/2026-05-11-cyaward-design.md) section 2

**User-confirmed scope adjustments:**
- Reuse Phase 1 fetch wrappers (Baseball-Reference still blocked, doesn't matter — bWAR isn't in feature set)
- Commit only `data/predictions/2026-MM-DD.parquet` (small daily files for sparkline); `data/raw/` stays git-ignored
- Late-season features stay 0-fill (training data also has them 0-filled)
- Local MVP first; GitHub Actions deployment is the last 2 tasks
- Include a manual sanity-check task before automating

---

## File Structure

```
src/
├── projector.py              # NEW: PaceProjector (Stage A)
├── eligibility.py            # NEW: SP/RP dynamic IP threshold
├── ranker.py                 # NEW: rank_today() orchestrator
├── render.py                 # NEW: Jinja2 HTML renderer
├── weekly_report.py          # NEW: weekly markdown generator
└── cli/
    ├── daily.py              # NEW: fetch → rank → save → render
    └── weekly.py             # NEW: aggregate week → emit markdown

templates/
└── dashboard.html.j2         # NEW: Jinja2 template for site/index.html

site/                         # GENERATED daily, committed
├── index.html
└── style.css                 # NEW: minimal CSS, committed

reports/
├── backtest_v1.md            # (existing) Phase 1
└── 2026-Wxx.md               # GENERATED weekly

data/
└── predictions/              # NEW: daily prediction parquets, committed
    └── 2026-MM-DD.parquet

.github/workflows/            # NEW
├── daily.yml
└── weekly.yml

tests/
├── test_projector.py         # NEW
├── test_eligibility.py       # NEW
├── test_ranker.py            # NEW
├── test_render.py            # NEW
└── test_weekly_report.py     # NEW
```

**Decomposition rationale:**
- `projector.py` is pure stat-projection logic (no I/O, no model). Easy to TDD with synthetic data.
- `eligibility.py` is pure date-arithmetic logic for the dynamic IP threshold. Separate from projector for clarity.
- `ranker.py` is the orchestrator: calls fetch, applies projector + eligibility, calls model.predict, sorts. Tests mock fetch and model.
- `render.py` and `weekly_report.py` are presentation only. Take DataFrames in, produce strings out.
- CLIs are thin orchestrators (parse args, glue modules, write to disk, commit).
- `templates/dashboard.html.j2` is the Jinja2 template — keep separate from render.py so HTML/CSS edits don't touch Python.

---

## Task 0: Project skeleton extension

**Files:**
- Modify: `.gitignore`
- Modify: `requirements.txt`
- Create: `data/predictions/.gitkeep`, `site/.gitkeep`, `templates/.gitkeep`
- Create: `src/cli/daily.py` (empty), `src/cli/weekly.py` (empty) — placeholder files only

- [ ] **Step 1: Add Jinja2 to requirements.txt**

Append to `/Users/ccli/Downloads/cyaward-claude/requirements.txt`:
```
jinja2>=3.1
```

- [ ] **Step 2: Update .gitignore**

The current `.gitignore` has `data/raw/` ignored. Phase 2 wants `data/predictions/` committed. Verify nothing extra needs changing. Read `.gitignore`; expected: `data/raw/` line is present; no change needed for `data/predictions/`.

If `site/` is currently ignored (unlikely but check), remove it — we want to commit `site/index.html`.

- [ ] **Step 3: Create directory placeholders**

```bash
mkdir -p data/predictions site templates .github/workflows
touch data/predictions/.gitkeep site/.gitkeep templates/.gitkeep
```

- [ ] **Step 4: Install new dep**

```bash
source .venv/bin/activate && pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt data/predictions/.gitkeep site/.gitkeep templates/.gitkeep .gitignore
git commit -m "chore(phase2): scaffold dirs + jinja2 dependency"
```

---

## Task 1: `src/eligibility.py` — Dynamic SP/RP IP filter

**Files:**
- Create: `src/eligibility.py`
- Test: `tests/test_eligibility.py`

Per spec section 2.3: `sp_min_ip = max(25, 162 * season_progress)`, `rp_min_ip = max(10, 60 * season_progress)`. SP if `GS/G > 0.5`, else RP.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eligibility.py
from datetime import date
import pandas as pd
import pytest
from src import eligibility


@pytest.fixture
def sample_pitchers():
    """3 pitchers: SP with enough IP, SP with too few IP, RP with enough IP."""
    return pd.DataFrame({
        "Name": ["Skubal_SP_OK", "Rookie_SP_LOW", "Clase_RP_OK"],
        "G":    [10, 8, 18],
        "GS":   [10, 8, 0],     # GS/G > 0.5 → SP; GS/G == 0 → RP
        "IP":   [60.0, 12.0, 18.0],
    })


def test_season_progress_midseason():
    # 2026 season: 3/26 to 9/27 = 183 days
    today = date(2026, 5, 12)  # 47 days in
    pct = eligibility.season_progress(today)
    assert 0.20 < pct < 0.30


def test_season_progress_pre_season_returns_zero():
    today = date(2026, 2, 1)
    assert eligibility.season_progress(today) == 0.0


def test_season_progress_post_season_capped_at_one():
    today = date(2026, 11, 1)
    assert eligibility.season_progress(today) == 1.0


def test_eligibility_thresholds_at_may_12():
    today = date(2026, 5, 12)
    sp_min, rp_min = eligibility.thresholds(today)
    # season_progress ≈ 0.257, sp_min = max(25, 162*0.257) ≈ max(25, 41.6) = 41.6
    assert sp_min == pytest.approx(162 * eligibility.season_progress(today))
    assert sp_min >= 25
    # rp_min = max(10, 60*0.257) ≈ max(10, 15.4) = 15.4
    assert rp_min >= 10


def test_filter_eligible_keeps_sp_above_threshold(sample_pitchers):
    today = date(2026, 5, 12)
    out = eligibility.filter_eligible(sample_pitchers, today)
    names = set(out["Name"])
    assert "Skubal_SP_OK" in names      # 60 IP > 41.6
    assert "Rookie_SP_LOW" not in names  # 12 IP < 41.6
    assert "Clase_RP_OK" in names        # RP, 18 IP > 15.4


def test_filter_eligible_returns_empty_pre_season(sample_pitchers):
    today = date(2026, 3, 1)  # before SEASON_START
    out = eligibility.filter_eligible(sample_pitchers, today)
    assert len(out) == 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_eligibility.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.eligibility`.

- [ ] **Step 3: Implement `src/eligibility.py`**

```python
# src/eligibility.py
"""Dynamic SP/RP IP eligibility thresholds for live mid-season ranking.

Per spec section 2.3: thresholds scale linearly with season progress.
Pre-season returns 0 progress (filter rejects everyone).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

SEASON_START = date(2026, 3, 26)
SEASON_END = date(2026, 9, 27)
SEASON_LENGTH_DAYS = (SEASON_END - SEASON_START).days  # 185

# Full-season IP floors (per spec)
SP_FULL_SEASON_IP = 162
RP_FULL_SEASON_IP = 60

# Minimum IP at any point in season (avoid empty rankings in early April)
SP_FLOOR_IP = 25
RP_FLOOR_IP = 10


def season_progress(today: date) -> float:
    """Fraction of regular season elapsed, clipped to [0, 1]."""
    if today < SEASON_START:
        return 0.0
    if today >= SEASON_END:
        return 1.0
    return (today - SEASON_START).days / SEASON_LENGTH_DAYS


def thresholds(today: date) -> tuple[float, float]:
    """Return (sp_min_ip, rp_min_ip) for ranking eligibility."""
    pct = season_progress(today)
    sp_min = max(SP_FLOOR_IP, SP_FULL_SEASON_IP * pct)
    rp_min = max(RP_FLOOR_IP, RP_FULL_SEASON_IP * pct)
    return sp_min, rp_min


def filter_eligible(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Filter a FanGraphs-shaped pitching frame to ranking-eligible pitchers.

    Pre-season → empty DataFrame.
    Otherwise:
        SP (GS/G > 0.5) require IP >= sp_min
        RP (GS/G <= 0.5) require IP >= rp_min
    """
    if today < SEASON_START:
        return df.iloc[0:0].copy()
    sp_min, rp_min = thresholds(today)
    is_sp = df["GS"] / df["G"] > 0.5
    keep_sp = is_sp & (df["IP"] >= sp_min)
    keep_rp = (~is_sp) & (df["IP"] >= rp_min)
    return df[keep_sp | keep_rp].copy().reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `source .venv/bin/activate && pytest tests/test_eligibility.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/eligibility.py tests/test_eligibility.py
git commit -m "feat(phase2): dynamic SP/RP IP eligibility filter"
```

---

## Task 2: `src/projector.py` — Pace × Remaining projector

**Files:**
- Create: `src/projector.py`
- Test: `tests/test_projector.py`

Stage A: project current cumulative stats to full-season equivalents. Per spec section 2.2.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_projector.py
from datetime import date
import pandas as pd
import pytest
from src import projector


@pytest.fixture
def current_stats_midseason():
    """A pitcher with ~40 IP at season_progress=0.25 (should project to ~160 IP)."""
    return pd.DataFrame({
        "Name":    ["Skubal", "Skenes"],
        "Team":    ["DET", "PIT"],
        "G":       [8, 7],
        "GS":      [8, 7],
        "IP":      [50.0, 40.0],
        "K":       [60, 55],
        "BB":      [10, 8],
        "ERA":     [2.50, 1.80],
        "fWAR":    [1.5, 1.2],
        "W":       [4, 3],
        "L":       [1, 2],
        "WHIP":    [0.95, 0.88],
        "CG":      [0, 0],
        "ShO":     [0, 0],
        "SV":      [0, 0],
        "FIP":     [2.40, 2.10],
        "xFIP":    [2.80, 2.50],
        "K-BB%":   [25.0, 24.5],
        "ERA-":    [60, 45],
        "FIP-":    [62, 55],
        "xERA":    [2.85, 2.20],
        "xwOBA_against": [0.255, 0.240],
        "Stuff+":  [115, 130],
        "Location+": [105, 102],
        "Barrel%": [5.5, 4.0],
        "HardHit%": [33.0, 30.0],
        "RS_per_9": [4.8, 5.2],
    })


def test_project_scales_ip_to_full_season(current_stats_midseason):
    today = date(2026, 5, 12)  # ~25% of season
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    # 50 IP at 25% → ~200 IP projected
    skubal_proj = proj[proj["Name"] == "Skubal"].iloc[0]
    assert 150 < skubal_proj["IP"] < 250


def test_project_preserves_rate_stats(current_stats_midseason):
    today = date(2026, 5, 12)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    skubal_orig = current_stats_midseason[current_stats_midseason["Name"] == "Skubal"].iloc[0]
    skubal_proj = proj[proj["Name"] == "Skubal"].iloc[0]
    # ERA, FIP, xERA, ERA-, Stuff+ are rate/index stats — should NOT scale
    assert skubal_proj["ERA"] == pytest.approx(skubal_orig["ERA"])
    assert skubal_proj["xERA"] == pytest.approx(skubal_orig["xERA"])
    assert skubal_proj["Stuff+"] == pytest.approx(skubal_orig["Stuff+"])


def test_project_scales_counting_stats(current_stats_midseason):
    today = date(2026, 5, 12)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    skubal_orig = current_stats_midseason[current_stats_midseason["Name"] == "Skubal"].iloc[0]
    skubal_proj = proj[proj["Name"] == "Skubal"].iloc[0]
    # K, BB, W, L, fWAR should scale up with IP
    ratio = skubal_proj["IP"] / skubal_orig["IP"]
    assert skubal_proj["K"] == pytest.approx(skubal_orig["K"] * ratio, rel=0.01)
    assert skubal_proj["fWAR"] == pytest.approx(skubal_orig["fWAR"] * ratio, rel=0.01)
    assert skubal_proj["W"] == pytest.approx(skubal_orig["W"] * ratio, rel=0.01)


def test_project_pre_season_returns_inputs_unchanged(current_stats_midseason):
    today = date(2026, 2, 1)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    # Pre-season: nothing to project, just pass through
    pd.testing.assert_frame_equal(
        proj.reset_index(drop=True),
        current_stats_midseason.reset_index(drop=True),
        check_dtype=False,
    )


def test_project_post_season_returns_inputs_unchanged(current_stats_midseason):
    today = date(2026, 11, 1)
    proj = projector.PaceProjector().project(current_stats_midseason, today)
    pd.testing.assert_frame_equal(
        proj.reset_index(drop=True),
        current_stats_midseason.reset_index(drop=True),
        check_dtype=False,
    )
```

- [ ] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_projector.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/projector.py`**

```python
# src/projector.py
"""Stage A: pace-based season projector.

Scales current cumulative stats (W, L, K, BB, IP, fWAR, ...) by the
inverse of season progress, leaving rate stats (ERA, FIP, xERA, ERA-,
FIP-, K-BB%, Stuff+, Location+, Barrel%, HardHit%, xwOBA_against, RS_per_9,
WHIP) unchanged.

Pre-season / post-season: pass through unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from src.eligibility import season_progress, SEASON_START, SEASON_END

# Stats that scale linearly with playing time
COUNTING_STATS = ["IP", "K", "BB", "W", "L", "fWAR", "CG", "ShO", "SV"]
# Stats that are already rates/ratios/indices — never scale them
RATE_STATS = ["ERA", "FIP", "xFIP", "xERA", "ERA-", "FIP-", "K-BB%",
              "WHIP", "Stuff+", "Location+", "Barrel%", "HardHit%",
              "xwOBA_against", "RS_per_9"]


class Projector(ABC):
    """Abstract interface (reserved for v3 Marcel-style projector)."""

    @abstractmethod
    def project(self, current: pd.DataFrame, asof_date: date) -> pd.DataFrame: ...


class PaceProjector(Projector):
    """Linear pace projection: IP_full = IP_current / season_progress."""

    def project(self, current: pd.DataFrame, asof_date: date) -> pd.DataFrame:
        # Pre-season or post-season: nothing to project
        if asof_date < SEASON_START or asof_date >= SEASON_END:
            return current.copy()

        pct = season_progress(asof_date)
        if pct <= 0:
            return current.copy()

        scale = 1.0 / pct  # e.g. 25% → multiply by 4
        out = current.copy()
        for col in COUNTING_STATS:
            if col in out.columns:
                out[col] = out[col] * scale
        # Rate stats: leave alone
        return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `source .venv/bin/activate && pytest tests/test_projector.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/projector.py tests/test_projector.py
git commit -m "feat(phase2): pace-based season projector (Stage A)"
```

---

## Task 3: `src/ranker.py` — Live ranking orchestrator

**Files:**
- Create: `src/ranker.py`
- Test: `tests/test_ranker.py`

`rank_today(asof_date)` — fetches current 2026 data, applies eligibility filter, projects, builds features, calls trained model, returns sorted ranking DataFrame.

- [ ] **Step 1: Write failing test**

```python
# tests/test_ranker.py
from datetime import date
from unittest.mock import patch
import pandas as pd
import pytest
import numpy as np
from src import ranker, config


@pytest.fixture
def mock_current_data(fake_fangraphs_df):
    """Reuse fake_fangraphs_df from conftest; rename to canonical names."""
    df = fake_fangraphs_df.rename(columns={
        "WAR": "fWAR", "SO": "K", "xwOBA": "xwOBA_against", "RS/9": "RS_per_9"
    }).copy()
    df["year"] = 2026
    # Bump IP to pass mid-season threshold
    df.loc[:, "IP"] = [80.0, 75.0, 70.0]
    df.loc[:, "G"] = [12, 11, 10]
    df.loc[:, "GS"] = [12, 11, 10]
    return df


@pytest.fixture
def mock_standings_2026():
    return pd.DataFrame({
        "Team": ["DET", "PIT", "OAK"],
        "year": [2026, 2026, 2026],
        "team_winning_pct": [0.580, 0.460, 0.420],
    })


def test_rank_today_returns_expected_schema(mock_current_data, mock_standings_2026):
    today = date(2026, 5, 12)
    with patch("src.ranker.fetch.get_fangraphs_pitching", return_value=mock_current_data), \
         patch("src.ranker.fetch.get_team_records", return_value=mock_standings_2026):
        result = ranker.rank_today(today)
    # Required columns
    assert "pitcher_name" in result.columns
    assert "Team" in result.columns
    assert "league" in result.columns
    assert "predicted_vote_share" in result.columns
    assert "predicted_rank_in_league" in result.columns
    assert "current_IP" in result.columns
    assert "current_ERA" in result.columns
    assert "current_fWAR" in result.columns
    assert "proj_IP" in result.columns


def test_rank_today_sorts_descending_within_league(mock_current_data, mock_standings_2026):
    today = date(2026, 5, 12)
    with patch("src.ranker.fetch.get_fangraphs_pitching", return_value=mock_current_data), \
         patch("src.ranker.fetch.get_team_records", return_value=mock_standings_2026):
        result = ranker.rank_today(today)
    for league, grp in result.groupby("league"):
        shares = grp.sort_values("predicted_rank_in_league")["predicted_vote_share"].values
        # rank 1 should have highest predicted share within league
        assert all(shares[i] >= shares[i+1] for i in range(len(shares)-1))


def test_rank_today_pre_season_returns_empty(mock_current_data, mock_standings_2026):
    today = date(2026, 2, 1)
    with patch("src.ranker.fetch.get_fangraphs_pitching", return_value=mock_current_data), \
         patch("src.ranker.fetch.get_team_records", return_value=mock_standings_2026):
        result = ranker.rank_today(today)
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_ranker.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/ranker.py`**

```python
# src/ranker.py
"""Phase 2 orchestrator: fetch current 2026 data → project → predict → rank.

Reuses Phase 1 fetch wrappers (works for any year), the trained GBR model
artifact, and features.build_features() — fed an empty awards DataFrame
since there's no 2026 voting yet (we only need feature columns).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src import fetch, features, voter_model, projector, eligibility, config


def _empty_awards() -> pd.DataFrame:
    """Awards frame with the canonical schema but no rows.

    Used in inference because 2026 voting hasn't happened yet. Every pitcher
    will get vote_share=0 / was_winner=0 after the left-join inside
    build_features — fine for prediction since we ignore those label cols.
    """
    return pd.DataFrame(columns=[
        "year", "league", "playerID", "pitcher_name",
        "pointsWon", "vote_share", "was_winner", "award",
    ])


def rank_today(asof_date: date, year: int = 2026) -> pd.DataFrame:
    """Produce ranked AL+NL Top-N DataFrame for the given as-of date.

    Steps: fetch → eligibility filter → pace projection → feature engineering
           → model.predict → sort by predicted_vote_share desc within league.
    """
    # Pre-season → empty ranking
    if asof_date < eligibility.SEASON_START:
        return pd.DataFrame()

    fg = fetch.get_fangraphs_pitching(year)
    bref = fetch.get_bref_pitching(year)
    standings = fetch.get_team_records(year)

    # Eligibility: dynamic IP threshold by season progress
    eligible = eligibility.filter_eligible(fg, asof_date)
    if eligible.empty:
        return pd.DataFrame()

    # Pace projection (counting stats scale; rates pass through)
    projected = projector.PaceProjector().project(eligible, asof_date)

    # Build features (awards is empty; ok for inference)
    awards = _empty_awards()
    feature_df = features.build_features(year, projected, bref, standings, awards)

    # Predict
    model = voter_model.load_model(config.GBR_MODEL_PATH)
    feature_df["predicted_vote_share"] = voter_model.predict(model, feature_df)

    # Rank within league
    feature_df = feature_df.sort_values(
        ["league", "predicted_vote_share"], ascending=[True, False]
    )
    feature_df["predicted_rank_in_league"] = (
        feature_df.groupby("league")["predicted_vote_share"].rank(
            method="min", ascending=False
        ).astype(int)
    )

    # Pack the output schema: current + projected key stats + prediction
    # Pull "current" stats by re-merging eligible's original cumulative values
    cur = eligible[["Name", "IP", "ERA", "fWAR", "xERA"]].rename(columns={
        "IP": "current_IP", "ERA": "current_ERA",
        "fWAR": "current_fWAR", "xERA": "current_xERA",
    })
    out = feature_df.merge(cur, left_on="pitcher_name", right_on="Name", how="left")

    # Projected key stats
    proj_key = projected[["Name", "IP", "ERA", "fWAR"]].rename(columns={
        "IP": "proj_IP", "ERA": "proj_ERA", "fWAR": "proj_fWAR",
    })
    out = out.merge(proj_key, left_on="pitcher_name", right_on="Name",
                    how="left", suffixes=("", "_proj_drop"))
    out = out.drop(columns=[c for c in out.columns if c.endswith("_proj_drop") or c == "Name"],
                   errors="ignore")

    return out[[
        "pitcher_name", "Team", "league", "predicted_rank_in_league",
        "predicted_vote_share",
        "current_IP", "current_ERA", "current_xERA", "current_fWAR",
        "proj_IP", "proj_ERA", "proj_fWAR",
    ]].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `source .venv/bin/activate && pytest tests/test_ranker.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ranker.py tests/test_ranker.py
git commit -m "feat(phase2): live ranker orchestrator (fetch → project → predict)"
```

---

## Task 4: Local sanity check (real 2026 data)

**Files:** none (manual run)

Before building rendering / actions, manually verify `rank_today(date(2026, 5, 12))` returns a reasonable AL+NL Top 10 with real FanGraphs data.

- [ ] **Step 1: Run ranker against live data**

```bash
source .venv/bin/activate && python -c "
from datetime import date
from src import ranker
df = ranker.rank_today(date(2026, 5, 12))
print(f'Rows: {len(df)}')
print()
print('AL Top 10:')
print(df[df.league=='AL'].head(10)[['predicted_rank_in_league','pitcher_name','Team','current_IP','current_ERA','current_fWAR','predicted_vote_share']].to_string(index=False))
print()
print('NL Top 10:')
print(df[df.league=='NL'].head(10)[['predicted_rank_in_league','pitcher_name','Team','current_IP','current_ERA','current_fWAR','predicted_vote_share']].to_string(index=False))
"
```

Expected: ~200-400 eligible pitchers; Top 10 of each league shown with sensible names (current 2026 elite starters) and predicted_vote_share between ~0.05 and ~0.5 (early-season uncertainty caps top values lower than late-season).

**STOP and report** if:
- 0 rows returned
- Top 10 contains obvious junk (no-name pitcher with 1 IP and 0.00 ERA topping the list)
- FanGraphs API throws an error
- predicted_vote_share is NaN

If output looks reasonable, proceed to Task 5. **Don't commit anything** in this task — it's purely a sanity check.

---

## Task 5: `src/render.py` — Jinja2 HTML renderer

**Files:**
- Create: `templates/dashboard.html.j2`
- Create: `site/style.css`
- Create: `src/render.py`
- Test: `tests/test_render.py`

Vanilla HTML + minimal CSS. Two columns (AL / NL) of 10 cards each. Each card shows rank, name, team, predicted share %, current IP/ERA/fWAR, projected IP/ERA/fWAR.

- [ ] **Step 1: Create `templates/dashboard.html.j2`**

```jinja2
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Cy Young Tracker — {{ asof_date }}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>MLB Cy Young Tracker</h1>
        <p class="subtitle">Predicted BBWAA vote shares as of <strong>{{ asof_date }}</strong></p>
        <p class="meta">Updated {{ generated_at }} · {{ total_eligible }} eligible pitchers</p>
    </header>

    <main>
        <section class="league" id="al">
            <h2>American League — Top 10</h2>
            <ol class="rankings">
                {% for p in al_top10 %}
                <li class="card">
                    <div class="card-header">
                        <span class="rank">#{{ p.predicted_rank_in_league }}</span>
                        <span class="name">{{ p.pitcher_name }}</span>
                        <span class="team">{{ p.Team }}</span>
                    </div>
                    <div class="share">{{ "%.1f" | format(p.predicted_vote_share * 100) }}%</div>
                    <div class="stats">
                        <span title="Current IP">IP {{ "%.1f" | format(p.current_IP) }}</span>
                        <span title="Current ERA">ERA {{ "%.2f" | format(p.current_ERA) }}</span>
                        <span title="Current fWAR">fWAR {{ "%.1f" | format(p.current_fWAR) }}</span>
                    </div>
                    <div class="stats projected">
                        <span title="Projected IP">→ IP {{ "%.0f" | format(p.proj_IP) }}</span>
                        <span title="Projected fWAR">fWAR {{ "%.1f" | format(p.proj_fWAR) }}</span>
                    </div>
                </li>
                {% endfor %}
            </ol>
        </section>

        <section class="league" id="nl">
            <h2>National League — Top 10</h2>
            <ol class="rankings">
                {% for p in nl_top10 %}
                <li class="card">
                    <div class="card-header">
                        <span class="rank">#{{ p.predicted_rank_in_league }}</span>
                        <span class="name">{{ p.pitcher_name }}</span>
                        <span class="team">{{ p.Team }}</span>
                    </div>
                    <div class="share">{{ "%.1f" | format(p.predicted_vote_share * 100) }}%</div>
                    <div class="stats">
                        <span title="Current IP">IP {{ "%.1f" | format(p.current_IP) }}</span>
                        <span title="Current ERA">ERA {{ "%.2f" | format(p.current_ERA) }}</span>
                        <span title="Current fWAR">fWAR {{ "%.1f" | format(p.current_fWAR) }}</span>
                    </div>
                    <div class="stats projected">
                        <span title="Projected IP">→ IP {{ "%.0f" | format(p.proj_IP) }}</span>
                        <span title="Projected fWAR">fWAR {{ "%.1f" | format(p.proj_fWAR) }}</span>
                    </div>
                </li>
                {% endfor %}
            </ol>
        </section>
    </main>

    <footer>
        <p>Model: Phase 1 GBR (trained on 2015–2025 BBWAA voting · MAE 0.0076 LOOCV).
           Source: <a href="https://github.com/ccli-taipower/cyaward-claude">github.com/ccli-taipower/cyaward-claude</a>.</p>
    </footer>
</body>
</html>
```

- [ ] **Step 2: Create `site/style.css`**

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #fafafa; color: #222; line-height: 1.5; padding: 20px; max-width: 1200px; margin: 0 auto; }
header { text-align: center; margin-bottom: 32px; }
header h1 { font-size: 2rem; margin-bottom: 8px; }
header .subtitle { color: #555; }
header .meta { color: #999; font-size: 0.85rem; margin-top: 4px; }
main { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 768px) { main { grid-template-columns: 1fr; } }
.league h2 { font-size: 1.2rem; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #ddd; }
.rankings { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.card { background: white; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px 16px; }
.card-header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px; }
.rank { font-weight: bold; color: #888; min-width: 32px; }
.name { font-weight: 600; flex: 1; }
.team { color: #666; font-size: 0.9rem; }
.share { font-size: 1.4rem; font-weight: bold; color: #1976d2; margin-bottom: 6px; }
.stats { display: flex; gap: 12px; font-size: 0.85rem; color: #555; }
.stats.projected { color: #888; font-style: italic; margin-top: 4px; }
footer { margin-top: 48px; text-align: center; color: #888; font-size: 0.85rem; }
footer a { color: #1976d2; }
```

- [ ] **Step 3: Write failing test**

```python
# tests/test_render.py
from datetime import date
from pathlib import Path
import pandas as pd
import pytest
from src import render


@pytest.fixture
def sample_ranking():
    """20 pitchers, 10 per league, with all required schema columns."""
    rows = []
    for league in ["AL", "NL"]:
        for i in range(10):
            rows.append({
                "pitcher_name": f"Pitcher_{league}_{i+1}",
                "Team": "DET" if league == "AL" else "PIT",
                "league": league,
                "predicted_rank_in_league": i + 1,
                "predicted_vote_share": 0.5 - i * 0.04,
                "current_IP": 60.0 - i,
                "current_ERA": 2.0 + i * 0.1,
                "current_xERA": 2.2 + i * 0.1,
                "current_fWAR": 2.5 - i * 0.2,
                "proj_IP": 200.0 - i * 5,
                "proj_ERA": 2.0 + i * 0.1,
                "proj_fWAR": 6.0 - i * 0.4,
            })
    return pd.DataFrame(rows)


def test_render_writes_html_with_expected_content(sample_ranking, tmp_path):
    out_path = tmp_path / "index.html"
    render.render_dashboard(
        ranking=sample_ranking,
        asof_date=date(2026, 5, 12),
        out_path=out_path,
    )
    text = out_path.read_text()
    assert "MLB Cy Young Tracker" in text
    assert "American League" in text
    assert "National League" in text
    assert "Pitcher_AL_1" in text  # top AL pitcher
    assert "Pitcher_NL_1" in text  # top NL pitcher
    assert "2026-05-12" in text


def test_render_handles_empty_ranking(tmp_path):
    out_path = tmp_path / "index.html"
    render.render_dashboard(
        ranking=pd.DataFrame(columns=[
            "pitcher_name", "Team", "league", "predicted_rank_in_league",
            "predicted_vote_share", "current_IP", "current_ERA", "current_xERA",
            "current_fWAR", "proj_IP", "proj_ERA", "proj_fWAR",
        ]),
        asof_date=date(2026, 3, 1),
        out_path=out_path,
    )
    text = out_path.read_text()
    assert "MLB Cy Young Tracker" in text
    # Empty league section is OK — page still renders
```

- [ ] **Step 4: Run test to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Implement `src/render.py`**

```python
# src/render.py
"""Render the daily AL+NL Top 10 dashboard from a ranking DataFrame.

Uses Jinja2 + the template in templates/dashboard.html.j2.
Output goes to site/index.html (so it's served by GitHub Pages).
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.config import ROOT

TEMPLATES_DIR = ROOT / "templates"
TOP_N_PER_LEAGUE = 10


def render_dashboard(
    ranking: pd.DataFrame,
    asof_date: date,
    out_path: Path | str,
) -> None:
    """Render the ranking DataFrame to a static HTML file."""
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
    template = env.get_template("dashboard.html.j2")

    if ranking.empty:
        al_top10 = []
        nl_top10 = []
        total = 0
    else:
        al_top10 = ranking[ranking["league"] == "AL"].head(TOP_N_PER_LEAGUE).to_dict("records")
        nl_top10 = ranking[ranking["league"] == "NL"].head(TOP_N_PER_LEAGUE).to_dict("records")
        total = len(ranking)

    html = template.render(
        asof_date=asof_date.isoformat(),
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        total_eligible=total,
        al_top10=al_top10,
        nl_top10=nl_top10,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
```

- [ ] **Step 6: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_render.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add templates/dashboard.html.j2 site/style.css src/render.py tests/test_render.py
git commit -m "feat(phase2): jinja2 dashboard renderer + minimal CSS"
```

---

## Task 6: `src/cli/daily.py` — Daily pipeline CLI

**Files:**
- Create: `src/cli/daily.py`
- Test: `tests/test_cli_daily.py`

Glue: parse `--date YYYY-MM-DD` (default = today UTC), call `rank_today`, save predictions parquet, call `render_dashboard`.

- [ ] **Step 1: Write failing test (mocks rank_today)**

```python
# tests/test_cli_daily.py
from datetime import date
from unittest.mock import patch
import pandas as pd
import pytest
from src.cli import daily


@pytest.fixture
def mock_ranking():
    rows = []
    for league in ["AL", "NL"]:
        for i in range(10):
            rows.append({
                "pitcher_name": f"P_{league}_{i+1}",
                "Team": "DET" if league == "AL" else "PIT",
                "league": league,
                "predicted_rank_in_league": i + 1,
                "predicted_vote_share": 0.5 - i * 0.04,
                "current_IP": 60.0 - i,
                "current_ERA": 2.0 + i * 0.1,
                "current_xERA": 2.2 + i * 0.1,
                "current_fWAR": 2.5 - i * 0.2,
                "proj_IP": 200.0 - i * 5,
                "proj_ERA": 2.0 + i * 0.1,
                "proj_fWAR": 6.0 - i * 0.4,
            })
    return pd.DataFrame(rows)


def test_daily_main_writes_predictions_and_site(tmp_path, monkeypatch, mock_ranking):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "predictions").mkdir(parents=True)
    (tmp_path / "site").mkdir(parents=True)
    (tmp_path / "templates").mkdir(parents=True)
    # Copy template into tmp path so render works
    import shutil
    from src.config import ROOT
    shutil.copy(ROOT / "templates" / "dashboard.html.j2",
                tmp_path / "templates" / "dashboard.html.j2")

    with patch("src.cli.daily.ranker.rank_today", return_value=mock_ranking), \
         patch("src.cli.daily.config.ROOT", tmp_path), \
         patch("src.cli.daily.PREDICTIONS_DIR", tmp_path / "data" / "predictions"), \
         patch("src.cli.daily.SITE_DIR", tmp_path / "site"):
        exit_code = daily.main(["--date", "2026-05-12"])

    assert exit_code == 0
    pred_path = tmp_path / "data" / "predictions" / "2026-05-12.parquet"
    site_path = tmp_path / "site" / "index.html"
    assert pred_path.exists()
    assert site_path.exists()
    pred = pd.read_parquet(pred_path)
    assert len(pred) == 20
```

- [ ] **Step 2: Run test to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_cli_daily.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/cli/daily.py`**

```python
# src/cli/daily.py
"""Daily Phase 2 pipeline: fetch → rank → save predictions → render dashboard."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from src import ranker, render, config

PREDICTIONS_DIR = config.DATA_DIR / "predictions"
SITE_DIR = config.ROOT / "site"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily Cy Young ranking pipeline")
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date.today(),
        help="As-of date (YYYY-MM-DD). Defaults to today UTC.",
    )
    args = parser.parse_args(argv)
    asof_date = args.date

    print(f"Running daily pipeline for {asof_date.isoformat()}...")

    try:
        ranking = ranker.rank_today(asof_date)
    except Exception as e:
        print(f"ERROR: ranker.rank_today failed: {e}", file=sys.stderr)
        return 1

    if ranking.empty:
        print(f"No eligible pitchers for {asof_date.isoformat()} (pre-season or empty data).")
        # Still render an "empty" dashboard so visitors see something sensible
        render.render_dashboard(ranking, asof_date, SITE_DIR / "index.html")
        return 0

    # Save daily predictions parquet (small; committed to git)
    pred_path = PREDICTIONS_DIR / f"{asof_date.isoformat()}.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(pred_path, index=False)
    print(f"Saved {len(ranking)} predictions -> {pred_path}")

    # Render HTML
    site_path = SITE_DIR / "index.html"
    site_path.parent.mkdir(parents=True, exist_ok=True)
    render.render_dashboard(ranking, asof_date, site_path)
    print(f"Rendered dashboard -> {site_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `source .venv/bin/activate && pytest tests/test_cli_daily.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run for real**

```bash
source .venv/bin/activate && python -m src.cli.daily --date 2026-05-12
```

Expected:
- `data/predictions/2026-05-12.parquet` created
- `site/index.html` created
- Open `site/index.html` in browser (`open site/index.html` on macOS) — should see AL Top 10 + NL Top 10 with sensible names + stats.

- [ ] **Step 6: Commit code + artifacts**

```bash
git add src/cli/daily.py tests/test_cli_daily.py data/predictions/2026-05-12.parquet site/index.html
git commit -m "feat(phase2): daily CLI; first 2026-05-12 ranking artifact"
```

---

## Task 7: `src/weekly_report.py` — Weekly markdown generator

**Files:**
- Create: `src/weekly_report.py`
- Test: `tests/test_weekly_report.py`

Aggregate the last 7 days of `data/predictions/*.parquet`, compute rank deltas, emit a markdown report.

- [ ] **Step 1: Write failing test**

```python
# tests/test_weekly_report.py
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import pytest
from src import weekly_report


@pytest.fixture
def fake_predictions_week(tmp_path):
    """Write 7 days of fake prediction parquets, with one pitcher rising and one falling."""
    pred_dir = tmp_path / "data" / "predictions"
    pred_dir.mkdir(parents=True)
    base = date(2026, 5, 5)  # Tuesday of week 19
    for i in range(7):
        day = base + timedelta(days=i)
        rows = []
        for league in ["AL", "NL"]:
            for j in range(10):
                # "Rising" pitcher (AL_riser): starts rank 8, climbs to rank 2 over the week
                # "Falling" pitcher (NL_faller): starts rank 1, falls to rank 5
                if league == "AL" and j == 1:
                    name = "AL_riser"
                    rank = 8 - i  # 8 → 2
                elif league == "NL" and j == 0:
                    name = "NL_faller"
                    rank = 1 + i // 2  # 1 → 4
                else:
                    name = f"P_{league}_{j+1}"
                    rank = j + 1
                rows.append({
                    "pitcher_name": name,
                    "Team": "DET",
                    "league": league,
                    "predicted_rank_in_league": rank,
                    "predicted_vote_share": 0.6 - rank * 0.05,
                    "current_IP": 60.0, "current_ERA": 2.5,
                    "current_xERA": 2.5, "current_fWAR": 2.0,
                    "proj_IP": 200.0, "proj_ERA": 2.5, "proj_fWAR": 6.0,
                })
        pd.DataFrame(rows).to_parquet(pred_dir / f"{day.isoformat()}.parquet", index=False)
    return pred_dir


def test_weekly_report_generates_markdown_with_sections(fake_predictions_week, tmp_path):
    out_path = tmp_path / "reports" / "2026-W19.md"
    weekly_report.generate_weekly_report(
        predictions_dir=fake_predictions_week,
        out_path=out_path,
        week_end=date(2026, 5, 11),
    )
    text = out_path.read_text()
    assert "# Cy Young Weekly" in text or "賽揚獎候選人週報" in text
    assert "AL" in text
    assert "NL" in text
    assert "Top 10" in text or "前 10" in text


def test_weekly_report_flags_rising_and_falling(fake_predictions_week, tmp_path):
    out_path = tmp_path / "reports" / "2026-W19.md"
    weekly_report.generate_weekly_report(
        predictions_dir=fake_predictions_week,
        out_path=out_path,
        week_end=date(2026, 5, 11),
    )
    text = out_path.read_text()
    # Rising pitcher (rank moved 8 → 2, +6) should appear
    assert "AL_riser" in text
    # Falling pitcher (rank moved 1 → 4, -3) should appear
    assert "NL_faller" in text
```

- [ ] **Step 2: Run tests to verify failure**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/weekly_report.py`**

```python
# src/weekly_report.py
"""Weekly markdown report aggregating the last 7 days of daily predictions.

Produces:
  - This week's AL/NL #1
  - Biggest rank movers (up + down)
  - New entrants to Top 10
  - Top 10 final tables
  - Methodology footnote
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

TOP_N = 10


def _load_week(predictions_dir: Path, week_end: date) -> dict[date, pd.DataFrame]:
    """Load the 7 daily prediction parquets ending at week_end (inclusive)."""
    out = {}
    for i in range(7):
        d = week_end - timedelta(days=6 - i)
        p = predictions_dir / f"{d.isoformat()}.parquet"
        if p.exists():
            out[d] = pd.read_parquet(p)
    return out


def _movers(week: dict[date, pd.DataFrame], league: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rank deltas between first and last day of the week."""
    days = sorted(week.keys())
    if len(days) < 2:
        empty = pd.DataFrame(columns=["pitcher_name", "delta", "rank_start", "rank_end"])
        return empty, empty
    start = week[days[0]]
    end = week[days[-1]]
    start_lg = start[start["league"] == league][["pitcher_name", "predicted_rank_in_league"]]
    start_lg = start_lg.rename(columns={"predicted_rank_in_league": "rank_start"})
    end_lg = end[end["league"] == league][["pitcher_name", "predicted_rank_in_league"]]
    end_lg = end_lg.rename(columns={"predicted_rank_in_league": "rank_end"})
    merged = end_lg.merge(start_lg, on="pitcher_name", how="left")
    merged["delta"] = merged["rank_start"] - merged["rank_end"]  # positive = improved
    risers = merged.sort_values("delta", ascending=False).head(5)
    fallers = merged.sort_values("delta", ascending=True).head(5)
    return risers, fallers


def _format_top_table(df: pd.DataFrame, league: str) -> list[str]:
    lg = df[df["league"] == league].head(TOP_N)
    lines = [
        f"### {league} Top {TOP_N}",
        "",
        "| Rank | Pitcher | Team | Vote share | IP | ERA | fWAR |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in lg.iterrows():
        lines.append(
            f"| {r.predicted_rank_in_league} | {r.pitcher_name} | {r.Team} | "
            f"{r.predicted_vote_share*100:.1f}% | {r.current_IP:.1f} | "
            f"{r.current_ERA:.2f} | {r.current_fWAR:.1f} |"
        )
    return lines


def generate_weekly_report(
    predictions_dir: Path,
    out_path: Path,
    week_end: date,
) -> None:
    """Generate a weekly markdown report."""
    week = _load_week(predictions_dir, week_end)
    if not week:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(f"# Cy Young Weekly — week ending {week_end}\n\n_No predictions available for this week._\n")
        return

    days = sorted(week.keys())
    latest = week[days[-1]]
    week_iso = week_end.isocalendar()
    week_label = f"{week_end.year}-W{week_iso[1]:02d}"

    lines = [
        f"# Cy Young Weekly — {week_label}",
        "",
        f"_Week ending {week_end.isoformat()} ({len(days)} daily snapshots loaded)_",
        "",
        "## This Week's #1s",
        "",
    ]
    for lg in ("AL", "NL"):
        top1 = latest[latest["league"] == lg].head(1)
        if not top1.empty:
            r = top1.iloc[0]
            lines.append(
                f"- **{lg}**: {r.pitcher_name} ({r.Team}) · "
                f"predicted vote share {r.predicted_vote_share*100:.1f}%"
            )
    lines.append("")

    for lg in ("AL", "NL"):
        risers, fallers = _movers(week, lg)
        lines.append(f"## {lg} Biggest Movers")
        lines.append("")
        lines.append("**Risers:**")
        for _, r in risers.iterrows():
            if pd.notna(r.delta) and r.delta > 0:
                lines.append(f"- 🟢 {r.pitcher_name}: rank {int(r.rank_start)} → {int(r.rank_end)} (+{int(r.delta)})")
        lines.append("")
        lines.append("**Fallers:**")
        for _, r in fallers.iterrows():
            if pd.notna(r.delta) and r.delta < 0:
                lines.append(f"- 🔴 {r.pitcher_name}: rank {int(r.rank_start)} → {int(r.rank_end)} ({int(r.delta)})")
        lines.append("")

    lines.append("## Top 10 — Latest Snapshot")
    lines.append("")
    for lg in ("AL", "NL"):
        lines.extend(_format_top_table(latest, lg))
        lines.append("")

    lines += [
        "## Methodology",
        "",
        "Model: Phase 1 GBR trained on 2015–2025 BBWAA Cy Young voting (MAE 0.0076 LOOCV).",
        "Predictions use pace × remaining projection of current 2026 stats.",
        "Source: <https://github.com/ccli-taipower/cyaward-claude>",
        "",
    ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && pytest tests/test_weekly_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/weekly_report.py tests/test_weekly_report.py
git commit -m "feat(phase2): weekly markdown report generator"
```

---

## Task 8: `src/cli/weekly.py` — Weekly CLI

**Files:**
- Create: `src/cli/weekly.py`

Glue: parse `--week-end YYYY-MM-DD` (default = last Sunday), call `generate_weekly_report`.

- [ ] **Step 1: Implement CLI**

```python
# src/cli/weekly.py
"""Weekly Phase 2 pipeline: load 7 days of predictions → emit markdown."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

from src import weekly_report, config

PREDICTIONS_DIR = config.DATA_DIR / "predictions"
REPORTS_DIR = config.REPORTS_DIR


def _default_week_end() -> date:
    """Most recent Sunday on or before today."""
    today = date.today()
    return today - timedelta(days=(today.weekday() + 1) % 7)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly Cy Young markdown report")
    parser.add_argument(
        "--week-end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=_default_week_end(),
        help="Last day of the week to report (YYYY-MM-DD). Defaults to last Sunday.",
    )
    args = parser.parse_args(argv)
    week_end = args.week_end
    week_iso = week_end.isocalendar()
    report_name = f"{week_end.year}-W{week_iso[1]:02d}.md"
    out_path = REPORTS_DIR / report_name

    print(f"Generating weekly report ending {week_end.isoformat()}...")
    weekly_report.generate_weekly_report(
        predictions_dir=PREDICTIONS_DIR,
        out_path=out_path,
        week_end=week_end,
    )
    print(f"Report written -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run for real (will work once we have ≥ 1 day of predictions)**

```bash
source .venv/bin/activate && python -m src.cli.weekly --week-end 2026-05-12
```

Expected: `reports/2026-W20.md` written. With only 1 prediction parquet available, the movers section will be empty but the top-10 table will populate.

- [ ] **Step 3: Commit**

```bash
git add src/cli/weekly.py reports/2026-W20.md
git commit -m "feat(phase2): weekly CLI; first 2026-W20 report"
```

---

## Task 9: `.github/workflows/daily.yml` — Daily Actions cron

**Files:**
- Create: `.github/workflows/daily.yml`

Per spec section 2.8: cron `0 11 * * *` (台灣 19:00). Steps: checkout → setup Python → restore cache → install deps → run daily CLI → commit if any artifacts changed → push.

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/daily.yml
name: Daily Cy Young Ranking

on:
  schedule:
    - cron: "0 11 * * *"   # 11:00 UTC = 19:00 Taiwan time, well after MLB game day
  workflow_dispatch:        # also support manual triggers

permissions:
  contents: write           # needed to push commits

jobs:
  rank-and-publish:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Cache pybaseball
        uses: actions/cache@v4
        with:
          path: ~/.pybaseball
          key: pybaseball-${{ runner.os }}

      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Run daily pipeline (with up to 3 retries)
        run: |
          source .venv/bin/activate
          for attempt in 1 2 3; do
            if python -m src.cli.daily; then
              exit 0
            fi
            echo "Attempt $attempt failed; sleeping..."
            sleep $((attempt * 60))
          done
          exit 1

      - name: Commit artifacts if changed
        run: |
          git config user.name  "cyaward-bot"
          git config user.email "cyaward-bot@users.noreply.github.com"
          git add data/predictions site/index.html
          if git diff --cached --quiet; then
            echo "No changes today."
          else
            git commit -m "data: daily ranking $(date -u +%Y-%m-%d)"
            git push
          fi
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/daily.yml
git commit -m "ci(phase2): daily Actions cron for ranking + dashboard"
```

---

## Task 10: `.github/workflows/weekly.yml` — Weekly Actions cron

**Files:**
- Create: `.github/workflows/weekly.yml`

Per spec section 2.8: cron `0 2 * * 1` (Mon UTC 02:00 = Taiwan Mon 10:00).

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/weekly.yml
name: Weekly Cy Young Report

on:
  schedule:
    - cron: "0 2 * * 1"    # Monday 02:00 UTC = Monday 10:00 Taiwan time
  workflow_dispatch:

permissions:
  contents: write

jobs:
  weekly-report:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Run weekly report
        run: |
          source .venv/bin/activate
          python -m src.cli.weekly

      - name: Commit report
        run: |
          git config user.name  "cyaward-bot"
          git config user.email "cyaward-bot@users.noreply.github.com"
          git add reports/
          if git diff --cached --quiet; then
            echo "No new report."
          else
            git commit -m "report: weekly $(date -u +%Y-W%V)"
            git push
          fi
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/weekly.yml
git commit -m "ci(phase2): weekly Actions cron for markdown report"
```

---

## Task 11: README — Phase 2 section

**Files:**
- Modify: `README.md`

Append a new "Phase 2" section under the existing "Status" / "Architecture" sections. Document the live dashboard, weekly report, deployment URL placeholder, and how to run locally.

- [ ] **Step 1: Read current README**

```bash
cat /Users/ccli/Downloads/cyaward-claude/README.md | head -10
```

- [ ] **Step 2: Update README**

Add a new section after the existing `## Architecture` section. The new section should contain:

```markdown
---

## Phase 2 — Live Dashboard

**Status:** Implemented; deployment to GitHub Pages pending.

The Phase 2 pipeline applies the validated Phase 1 GBR model to live 2026 stats and produces:

- **Daily HTML dashboard** at `site/index.html` (intended for GitHub Pages) with AL+NL Top 10.
- **Weekly markdown report** at `reports/2026-Wxx.md` (every Monday) with movers and Top 10 snapshot.

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

Two GitHub Actions workflows are committed but require manual enablement on the repo:

- `.github/workflows/daily.yml` — `cron: "0 11 * * *"` (台灣 19:00). Runs the daily pipeline; commits the new parquet + updated `site/index.html`.
- `.github/workflows/weekly.yml` — `cron: "0 2 * * 1"` (台灣 週一 10:00). Generates the weekly report.

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
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(phase2): document live dashboard + weekly report"
```

---

## Task 12: Final verification + tag

**Files:** none

- [ ] **Step 1: Run the full test suite**

```bash
source .venv/bin/activate && pytest -v
```

Expected: 41 Phase 1 tests + new Phase 2 tests, all green. New tests:
- 6 from `test_eligibility.py`
- 5 from `test_projector.py`
- 3 from `test_ranker.py`
- 2 from `test_render.py`
- 1 from `test_cli_daily.py`
- 2 from `test_weekly_report.py`
- Total new: ~19
- Total all: ~60

- [ ] **Step 2: Confirm DoD items from spec section 2.9**

1. `python -m src.cli.daily --date 2026-05-12` produces `site/index.html` + `data/predictions/2026-05-12.parquet` ✓ (verified in Task 6)
2. `python -m src.cli.weekly --week-end 2026-05-12` produces `reports/2026-Wxx.md` ✓ (verified in Task 8)
3. `.github/workflows/daily.yml` and `weekly.yml` exist; manual trigger via `workflow_dispatch` available ✓ (verified in Tasks 9–10; actual cron firing waits for first scheduled run on the remote)
4. GitHub Pages deployment — **deferred**; will be enabled after `main` has the first dashboard committed and the user enables Pages in repo settings (pointing to `/site` folder on `main` branch).
5. All Phase 2 tests green ✓
6. README Phase 2 section ✓

Items 3 (cron firing) and 4 (Pages enablement) cannot be self-verified — they require pushing to a GitHub remote and the repo owner enabling Pages. Note this in the report.

- [ ] **Step 3: Tag the milestone**

```bash
git tag -a phase2-complete -m "Phase 2 complete: live dashboard + weekly report + Actions cron

Daily pipeline: fetch 2026 FG/MLB Stats/Savant → pace projector → Phase 1 GBR predict → AL+NL Top 10 dashboard.
Weekly pipeline: aggregate 7 daily parquets → markdown report with movers + tables.

First artifact: 2026-05-12 ranking committed at <SHA>.
Pages deployment and first cron firing require remote push + repo settings.
"
```

- [ ] **Step 4: Final status report**

State explicitly:
- Phase 2 implementation complete locally.
- One real-data dashboard produced for 2026-05-12, visible at `site/index.html`.
- All Phase 2 unit tests passing.
- Next manual step: push `main` to GitHub remote and enable Pages in repo settings.
- Future autonomous step: daily/weekly crons fire once the workflows are visible on `main`.
