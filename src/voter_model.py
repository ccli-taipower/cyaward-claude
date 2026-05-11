"""Voter-share regression: GradientBoosting (main) + Ridge (baseline)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS


def train_gbr(training_df: pd.DataFrame) -> Pipeline:
    """GradientBoosting — primary model. Handles NaN via median imputer."""
    X = training_df[FEATURE_COLS]
    y = training_df["vote_share"]
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pipe


def train_ridge(training_df: pd.DataFrame) -> Pipeline:
    """Ridge baseline. Needs scaling; same imputer."""
    X = training_df[FEATURE_COLS]
    y = training_df["vote_share"]
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict and clip to [0, 1] (vote_share is bounded)."""
    raw = model.predict(X[FEATURE_COLS])
    return np.clip(raw, 0.0, 1.0)


def save_model(model: Pipeline, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> Pipeline:
    return joblib.load(path)


def train_calibrator(predicted_share: np.ndarray, was_winner: np.ndarray) -> IsotonicRegression:
    """Map predicted vote_share -> P(was_winner). Monotonic non-decreasing."""
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(predicted_share, was_winner)
    return cal
