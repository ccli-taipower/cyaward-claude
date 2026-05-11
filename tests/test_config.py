from src import config


def test_training_years_excludes_2020():
    assert 2020 not in config.TRAINING_YEARS
    assert set(config.TRAINING_YEARS) == {2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024}


def test_kpi_targets_match_spec():
    assert config.KPI_TARGETS["winner_hits_min"] == 14
    assert config.KPI_TARGETS["winner_hits_total"] == 18
    assert config.KPI_TARGETS["podium_overlap_avg_min"] == 2.0
    assert config.KPI_TARGETS["top10_overlap_avg_min"] == 7.0


def test_feature_cols_count():
    # 26 features: 10 traditional + 6 sabermetric + 6 statcast + 4 context
    assert len(config.FEATURE_COLS) == 26


def test_paths_resolve():
    assert config.TRAINING_PARQUET.name == "training_2015_2024.parquet"
    assert config.GBR_MODEL_PATH.name == "voter_model_gbr_v1.pkl"
