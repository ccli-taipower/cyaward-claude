"""Run leave-one-year-out CV + time-series split, write report, save calibrator."""
from __future__ import annotations

import sys
import pandas as pd

from src import backtest, voter_model, config


def main() -> int:
    if not config.TRAINING_PARQUET.exists():
        print(f"ERROR: {config.TRAINING_PARQUET} not found.", file=sys.stderr)
        return 1
    df = pd.read_parquet(config.TRAINING_PARQUET)
    print(f"Loaded {len(df)} training rows.")

    # Build truth frame from training_df (one row per pitcher per year)
    truth = df[["year", "league", "pitcher_name", "vote_share", "was_winner"]].rename(
        columns={"vote_share": "actual_vote_share"}
    )

    print(f"\nRunning LOOCV (GBR) over {len(config.TRAINING_YEARS)} years...")
    gbr_pred, gbr_oof = backtest.run_loocv(df, voter_model.train_gbr, config.TRAINING_YEARS)

    print("Running time-series split (GBR train 2015-2022 / val 2023 / test 2023)...")
    ts_pred = backtest.run_timesplit(
        df,
        voter_model.train_gbr,
        train_years=[y for y in config.TRAINING_YEARS if y <= 2022],
        test_year=2023,
    )

    print("Generating report ...")
    summary = backtest.generate_report(
        loocv_pred=gbr_pred,
        truth=truth,
        timesplit_pred=ts_pred,
        out_path=config.BACKTEST_REPORT_PATH,
        model_label="GradientBoostingRegressor v1",
    )

    print("\nTraining calibrator from GBR out-of-fold predictions ...")
    cal = voter_model.train_calibrator(
        gbr_oof["predicted_vote_share"].values,
        gbr_oof["was_winner"].values,
    )
    voter_model.save_model(cal, config.CALIBRATOR_PATH)
    print(f"  -> {config.CALIBRATOR_PATH}")

    # Also run Ridge for comparison and append a line to the report
    print("\nRunning LOOCV (Ridge) for baseline comparison ...")
    ridge_pred, _ = backtest.run_loocv(df, voter_model.train_ridge, config.TRAINING_YEARS)
    ridge_winner = backtest.winner_hits(ridge_pred, truth)
    with open(config.BACKTEST_REPORT_PATH, "a") as f:
        f.write(f"\n## Ridge Baseline (LOOCV)\n\n")
        f.write(f"Winner hits: **{ridge_winner} / {config.KPI_TARGETS['winner_hits_total']}** "
                f"(GBR: {summary['winner_hits']})\n")

    print(f"\n=== KPI summary (GBR primary) ===")
    print(f"  Winner hits:    {summary['winner_hits']} / {config.KPI_TARGETS['winner_hits_total']}")
    print(f"  Podium avg:     {summary['podium_avg']:.2f} / 3.0")
    print(f"  Top-10 avg:     {summary['top10_avg']:.2f} / 10.0")
    print(f"  Vote share MAE: {summary['mae']:.4f}")
    print(f"  Overall:        {'PASS' if summary['overall_pass'] else 'FAIL'}")
    print(f"\nReport saved -> {config.BACKTEST_REPORT_PATH}")

    return 0 if summary["overall_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
