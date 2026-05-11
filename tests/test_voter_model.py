import numpy as np
import pandas as pd
import pytest
from src import voter_model, config


@pytest.fixture
def synthetic_training_set():
    """Tiny synthetic dataset where vote_share strongly correlates with fWAR."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({col: rng.normal(0, 1, n) for col in config.FEATURE_COLS})
    df["vote_share"] = np.clip(0.05 * df["fWAR"] + rng.normal(0, 0.05, n), 0, 1)
    return df


def test_train_gbr_returns_fitted_pipeline(synthetic_training_set):
    pipe = voter_model.train_gbr(synthetic_training_set)
    X = synthetic_training_set[config.FEATURE_COLS].head(5)
    preds = pipe.predict(X)
    assert preds.shape == (5,)
    assert (preds >= -0.5).all() and (preds <= 1.5).all()  # sanity bounds


def test_train_ridge_returns_fitted_pipeline(synthetic_training_set):
    pipe = voter_model.train_ridge(synthetic_training_set)
    X = synthetic_training_set[config.FEATURE_COLS].head(5)
    preds = pipe.predict(X)
    assert preds.shape == (5,)


def test_save_and_load_model_roundtrip(tmp_path, synthetic_training_set):
    pipe = voter_model.train_gbr(synthetic_training_set)
    path = tmp_path / "test_model.pkl"
    voter_model.save_model(pipe, path)
    loaded = voter_model.load_model(path)
    X = synthetic_training_set[config.FEATURE_COLS].head(3)
    np.testing.assert_array_almost_equal(pipe.predict(X), loaded.predict(X))


def test_predict_clips_to_unit_interval(synthetic_training_set):
    pipe = voter_model.train_gbr(synthetic_training_set)
    X = synthetic_training_set[config.FEATURE_COLS].head(20)
    preds = voter_model.predict(pipe, X)
    assert (preds >= 0).all()
    assert (preds <= 1).all()


def test_train_calibrator_returns_isotonic():
    rng = np.random.default_rng(0)
    predicted_share = rng.uniform(0, 1, 100)
    # synthetic: high predicted share -> more likely winner
    was_winner = (predicted_share > 0.7).astype(int)
    cal = voter_model.train_calibrator(predicted_share, was_winner)
    # high share input -> high probability output
    assert cal.predict([0.9])[0] > cal.predict([0.1])[0]


def test_calibrator_output_in_unit_interval():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, 50)
    won = (pred > 0.6).astype(int)
    cal = voter_model.train_calibrator(pred, won)
    out = cal.predict([0.0, 0.3, 0.5, 0.8, 1.0])
    assert (out >= 0).all() and (out <= 1).all()
