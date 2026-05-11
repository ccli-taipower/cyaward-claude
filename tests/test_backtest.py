import pandas as pd
import pytest
from src import backtest


@pytest.fixture
def sample_predictions_and_truth():
    """One year, both leagues. Predictions ranked by predicted_vote_share desc within league."""
    pred = pd.DataFrame({
        "year":    [2024]*4 + [2024]*4,
        "league":  ["AL"]*4 + ["NL"]*4,
        "pitcher_name": ["Skubal", "Ragans", "Cole", "Crochet",
                         "Skenes",  "Wheeler", "Sale", "Webb"],
        "predicted_vote_share": [0.85, 0.55, 0.30, 0.20,
                                 0.78, 0.60, 0.50, 0.40],
    })
    truth = pd.DataFrame({
        "year":    [2024]*4 + [2024]*4,
        "league":  ["AL"]*4 + ["NL"]*4,
        "pitcher_name": ["Skubal", "Ragans", "Cole", "Lugo",       # AL: top1 match, top3 = 2/3 + Cole gives 3/3
                         "Sale",   "Skenes", "Wheeler", "Lopez"],  # NL: top1 mismatch, top3 set match
        "actual_vote_share": [0.95, 0.45, 0.20, 0.15,
                              0.65, 0.62, 0.50, 0.10],
        "was_winner": [1, 0, 0, 0, 1, 0, 0, 0],
    })
    return pred, truth


def test_winner_hits_count(sample_predictions_and_truth):
    pred, truth = sample_predictions_and_truth
    hits = backtest.winner_hits(pred, truth)
    # AL: predicted top1 = Skubal, actual winner = Skubal -> hit
    # NL: predicted top1 = Skenes, actual winner = Sale -> miss
    assert hits == 1


def test_podium_overlap_per_case(sample_predictions_and_truth):
    pred, truth = sample_predictions_and_truth
    overlaps = backtest.podium_overlap(pred, truth, top_n=3)
    # AL: pred top3 = {Skubal, Ragans, Cole}; actual top3 = {Skubal, Ragans, Cole} -> 3
    # NL: pred top3 = {Skenes, Wheeler, Sale}; actual top3 = {Sale, Skenes, Wheeler} -> 3
    assert sorted(overlaps) == [3, 3]


def test_top10_overlap(sample_predictions_and_truth):
    pred, truth = sample_predictions_and_truth
    overlaps = backtest.podium_overlap(pred, truth, top_n=4)  # using 4 since fixture has 4 each
    # AL: pred top4 = {Skubal, Ragans, Cole, Crochet}; actual top4 = {Skubal, Ragans, Cole, Lugo} -> 3
    # NL: pred top4 = {Skenes, Wheeler, Sale, Webb}; actual top4 = {Sale, Skenes, Wheeler, Lopez} -> 3
    assert sorted(overlaps) == [3, 3]


def test_vote_share_mae(sample_predictions_and_truth):
    pred, truth = sample_predictions_and_truth
    mae = backtest.vote_share_mae(pred, truth)
    assert mae > 0
    assert mae < 0.5
