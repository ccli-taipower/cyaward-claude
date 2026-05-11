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

    fg = fetch.get_fangraphs_pitching(year, force_refresh=True)
    bref = fetch.get_bref_pitching(year)
    standings = fetch.get_team_records(year, force_refresh=True)

    # Eligibility: dynamic IP threshold by season progress
    eligible = eligibility.filter_eligible(fg, asof_date)
    if eligible.empty:
        return pd.DataFrame()

    # Snapshot pre-projection current stats for display
    current_stats = eligible[["Name", "IP", "ERA", "fWAR", "xERA"]].rename(columns={
        "IP": "current_IP", "ERA": "current_ERA",
        "fWAR": "current_fWAR", "xERA": "current_xERA",
    }).copy()

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

    # Re-merge current + projected key stats for display
    out = feature_df.merge(current_stats, left_on="pitcher_name", right_on="Name", how="left")

    proj_key = projected[["Name", "IP", "ERA", "fWAR"]].rename(columns={
        "IP": "proj_IP", "ERA": "proj_ERA", "fWAR": "proj_fWAR",
    })
    out = out.merge(proj_key, left_on="pitcher_name", right_on="Name",
                    how="left", suffixes=("", "_proj_drop"))
    # Drop helper Name columns
    drop_cols = [c for c in out.columns if c == "Name" or c.endswith("_proj_drop")]
    out = out.drop(columns=drop_cols, errors="ignore")

    return out[[
        "pitcher_name", "Team", "league", "predicted_rank_in_league",
        "predicted_vote_share",
        "current_IP", "current_ERA", "current_xERA", "current_fWAR",
        "proj_IP", "proj_ERA", "proj_fWAR",
    ]].reset_index(drop=True)
