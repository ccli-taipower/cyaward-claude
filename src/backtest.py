"""Backtest framework: metrics, LOOCV, time-series split, report generation."""
from __future__ import annotations

import pandas as pd


def _predicted_top_n(pred: pd.DataFrame, n: int) -> pd.DataFrame:
    """For each (year, league), return top-n rows by predicted_vote_share."""
    return (
        pred.sort_values(["year", "league", "predicted_vote_share"], ascending=[True, True, False])
            .groupby(["year", "league"], group_keys=False)
            .head(n)
    )


def _actual_top_n(truth: pd.DataFrame, n: int) -> pd.DataFrame:
    return (
        truth.sort_values(["year", "league", "actual_vote_share"], ascending=[True, True, False])
             .groupby(["year", "league"], group_keys=False)
             .head(n)
    )


def winner_hits(pred: pd.DataFrame, truth: pd.DataFrame) -> int:
    """Count of (year, league) where predicted top-1 == actual winner."""
    pred_top1 = _predicted_top_n(pred, 1)[["year", "league", "pitcher_name"]]
    actual_winners = truth[truth["was_winner"] == 1][["year", "league", "pitcher_name"]]
    merged = pred_top1.merge(
        actual_winners,
        on=["year", "league", "pitcher_name"],
        how="inner",
    )
    return len(merged)


def podium_overlap(pred: pd.DataFrame, truth: pd.DataFrame, top_n: int = 3) -> list[int]:
    """List of overlap counts (one per year-league case)."""
    pt = _predicted_top_n(pred, top_n)
    at = _actual_top_n(truth, top_n)
    overlaps = []
    for (yr, lg), p_grp in pt.groupby(["year", "league"]):
        a_grp = at[(at["year"] == yr) & (at["league"] == lg)]
        ov = len(set(p_grp["pitcher_name"]) & set(a_grp["pitcher_name"]))
        overlaps.append(ov)
    return overlaps


def vote_share_mae(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    merged = pred.merge(truth, on=["year", "league", "pitcher_name"], how="inner")
    return (merged["predicted_vote_share"] - merged["actual_vote_share"]).abs().mean()


import numpy as np
from typing import Callable
from src.config import FEATURE_COLS
from src import voter_model


def run_loocv(
    training_df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame], object],
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-year-out CV.

    Returns:
        predictions: one row per held-out pitcher with predicted_vote_share
        oof_calibration: same rows but with was_winner — feeds calibrator training
    """
    pred_parts = []
    for held_out_year in years:
        train_mask = training_df["year"] != held_out_year
        test_mask = ~train_mask
        train = training_df[train_mask]
        test = training_df[test_mask].copy()

        model = model_factory(train)
        test["predicted_vote_share"] = voter_model.predict(model, test)
        pred_parts.append(test)

    pred_full = pd.concat(pred_parts, ignore_index=True)
    pred_out = pred_full[["year", "league", "pitcher_name", "predicted_vote_share"]].copy()
    oof_cal = pred_full[["predicted_vote_share", "was_winner"]].copy()
    return pred_out, oof_cal
