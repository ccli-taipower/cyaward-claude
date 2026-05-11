# src/render.py
"""Render the daily AL+NL Top 10 dashboard from a ranking DataFrame.

Uses Jinja2 + the template in templates/dashboard.html.j2.
Output goes to site/index.html (so it's served by GitHub Pages).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
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
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_eligible=total,
        al_top10=al_top10,
        nl_top10=nl_top10,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
