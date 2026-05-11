"""Train both voter models on the full training parquet, save pkls."""
from __future__ import annotations

import sys
import pandas as pd

from src import voter_model, config


def main() -> int:
    if not config.TRAINING_PARQUET.exists():
        print(f"ERROR: {config.TRAINING_PARQUET} not found. Run build_training_data first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(config.TRAINING_PARQUET)
    print(f"Loaded {len(df)} training rows from {config.TRAINING_PARQUET}")

    print("Training GradientBoosting ...")
    gbr = voter_model.train_gbr(df)
    voter_model.save_model(gbr, config.GBR_MODEL_PATH)
    print(f"  -> {config.GBR_MODEL_PATH}")

    print("Training Ridge ...")
    ridge = voter_model.train_ridge(df)
    voter_model.save_model(ridge, config.RIDGE_MODEL_PATH)
    print(f"  -> {config.RIDGE_MODEL_PATH}")

    print("\nTrain-set sanity check (GBR):")
    preds = voter_model.predict(gbr, df)
    print(f"  vote_share range predicted: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"  vote_share range actual:    [{df['vote_share'].min():.3f}, {df['vote_share'].max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
