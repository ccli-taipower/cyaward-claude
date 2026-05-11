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
