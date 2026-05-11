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
        Path(out_path).write_text(f"# Cy Young Weekly - week ending {week_end}\n\n_No predictions available for this week._\n")
        return

    days = sorted(week.keys())
    latest = week[days[-1]]
    week_iso = week_end.isocalendar()
    week_label = f"{week_end.year}-W{week_iso[1]:02d}"

    lines = [
        f"# Cy Young Weekly - {week_label}",
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
                f"- **{lg}**: {r.pitcher_name} ({r.Team}) - "
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
                lines.append(f"- {r.pitcher_name}: rank {int(r.rank_start)} -> {int(r.rank_end)} (+{int(r.delta)})")
        lines.append("")
        lines.append("**Fallers:**")
        for _, r in fallers.iterrows():
            if pd.notna(r.delta) and r.delta < 0:
                lines.append(f"- {r.pitcher_name}: rank {int(r.rank_start)} -> {int(r.rank_end)} ({int(r.delta)})")
        lines.append("")

    lines.append("## Top 10 - Latest Snapshot")
    lines.append("")
    for lg in ("AL", "NL"):
        lines.extend(_format_top_table(latest, lg))
        lines.append("")

    lines += [
        "## Methodology",
        "",
        "Model: Phase 1 GBR trained on 2015-2025 BBWAA Cy Young voting (MAE 0.0076 LOOCV).",
        "Predictions use pace x remaining projection of current 2026 stats.",
        "Source: <https://github.com/ccli-taipower/cyaward-claude>",
        "",
    ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
