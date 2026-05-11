# src/cli/build_training_data.py
"""Build the historical training parquet by looping TRAINING_YEARS."""
from __future__ import annotations

import sys
import pandas as pd

from src import fetch, features, config


def main() -> int:
    print(f"Building training data for years: {config.TRAINING_YEARS}")

    # Awards: fetch all years in one call (Lahman is a single table)
    awards = fetch.get_awards_history(config.TRAINING_YEARS)
    if awards.empty:
        print("ERROR: awards data is empty — Lahman snapshot may be stale", file=sys.stderr)
        return 1

    missing_years = set(config.TRAINING_YEARS) - set(awards["year"].unique())
    if missing_years:
        print(f"WARNING: awards missing for years {sorted(missing_years)} — those years' winners will be unknown",
              file=sys.stderr)

    parts = []
    for year in config.TRAINING_YEARS:
        print(f"  fetching {year} ...")
        fg = fetch.get_fangraphs_pitching(year)
        bref = fetch.get_bref_pitching(year)
        standings = fetch.get_team_records(year)
        rows = features.build_features(year, fg, bref, standings, awards)
        print(f"  -> {len(rows)} eligible pitcher-rows ({rows['was_winner'].sum()} winners)")
        parts.append(rows)

    full = pd.concat(parts, ignore_index=True)
    config.TRAINING_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(config.TRAINING_PARQUET, index=False)
    print(f"\nSaved {len(full)} rows -> {config.TRAINING_PARQUET}")
    print(f"Winners total: {full['was_winner'].sum()} (expected: {len(config.TRAINING_YEARS)*2} = years × leagues)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
