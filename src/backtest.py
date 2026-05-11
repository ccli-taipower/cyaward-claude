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


def run_timesplit(
    training_df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame], object],
    train_years: list[int],
    test_year: int,
) -> pd.DataFrame:
    """Single train/test split."""
    train = training_df[training_df["year"].isin(train_years)]
    test = training_df[training_df["year"] == test_year].copy()
    model = model_factory(train)
    test["predicted_vote_share"] = voter_model.predict(model, test)
    return test[["year", "league", "pitcher_name", "predicted_vote_share"]]


from pathlib import Path
import statistics
from src.config import KPI_TARGETS


def _outliers(pred: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    """Cases where predicted top1 != actual winner."""
    pred_top1 = _predicted_top_n(pred, 1)
    out = []
    for (yr, lg), grp in pred_top1.groupby(["year", "league"]):
        predicted = grp["pitcher_name"].iloc[0]
        actual = truth[(truth["year"] == yr) & (truth["league"] == lg) & (truth["was_winner"] == 1)]
        if actual.empty:
            continue
        actual_name = actual["pitcher_name"].iloc[0]
        if predicted != actual_name:
            out.append({"year": yr, "league": lg, "predicted": predicted, "actual": actual_name})
    return out


def generate_report(
    loocv_pred: pd.DataFrame,
    truth: pd.DataFrame,
    timesplit_pred: pd.DataFrame,
    out_path: "Path | str",
    model_label: str,
) -> dict:
    """Write backtest_v1.md and return KPI summary dict."""
    winner_count = winner_hits(loocv_pred, truth)
    podium = podium_overlap(loocv_pred, truth, top_n=3)
    top10 = podium_overlap(loocv_pred, truth, top_n=10)
    mae = vote_share_mae(loocv_pred, truth)
    outliers = _outliers(loocv_pred, truth)

    n_cases = KPI_TARGETS["winner_hits_total"]
    podium_avg = statistics.mean(podium) if podium else 0.0
    top10_avg = statistics.mean(top10) if top10 else 0.0

    t1_pass = winner_count >= KPI_TARGETS["winner_hits_min"]
    t2_pass = podium_avg >= KPI_TARGETS["podium_overlap_avg_min"]
    t3_pass = top10_avg >= KPI_TARGETS["top10_overlap_avg_min"]
    overall = t1_pass and t2_pass and t3_pass

    def status(p): return "PASS" if p else "FAIL"

    lines = [
        f"# Backtest Report — {model_label}",
        "",
        f"**Overall verdict:** {status(overall)}",
        "",
        "## KPI Summary (LOOCV)",
        "",
        "| Tier | Metric | Target | Result | Status |",
        "|---|---|---|---|---|",
        f"| Tier 1 | Winner hits | >= {KPI_TARGETS['winner_hits_min']} / {n_cases} | {winner_count} / {n_cases} | {status(t1_pass)} |",
        f"| Tier 2 | Podium overlap avg | >= {KPI_TARGETS['podium_overlap_avg_min']:.1f} / 3 | {podium_avg:.2f} / 3 | {status(t2_pass)} |",
        f"| Tier 3 | Top-10 overlap avg | >= {KPI_TARGETS['top10_overlap_avg_min']:.1f} / 10 | {top10_avg:.2f} / 10 | {status(t3_pass)} |",
        "",
        f"**Vote-share MAE (LOOCV):** {mae:.4f}",
        "",
        "## Outlier Cases (predicted top-1 != actual winner)",
        "",
    ]
    if outliers:
        lines.append("| Year | League | Predicted | Actual |")
        lines.append("|---|---|---|---|")
        for o in outliers:
            lines.append(f"| {o['year']} | {o['league']} | {o['predicted']} | {o['actual']} |")
    else:
        lines.append("_No outliers — all winners predicted correctly._")

    lines += [
        "",
        "## Time-Series Split Sanity (train 2015-2022 / val 2023 / test 2024)",
        "",
        f"Time-split predictions count: {len(timesplit_pred)}",
        "(See `models/voter_model_*_v1.pkl` for the trained models.)",
        "",
    ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))

    return {
        "winner_hits": winner_count,
        "podium_avg": podium_avg,
        "top10_avg": top10_avg,
        "mae": mae,
        "overall_pass": overall,
    }
