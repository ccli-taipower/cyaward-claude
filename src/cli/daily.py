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
