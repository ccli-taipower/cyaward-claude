# tests/conftest.py
import pandas as pd
import pytest


@pytest.fixture
def fake_fangraphs_df():
    """Mimics what pybaseball.pitching_stats returns. Just enough columns."""
    return pd.DataFrame({
        "Name":  ["Tarik Skubal", "Paul Skenes", "Bad Pitcher"],
        "Team":  ["DET", "PIT", "OAK"],
        "Age":   [27, 22, 30],
        "W":     [18, 11, 4],
        "L":     [4, 3, 13],
        "ERA":   [2.39, 1.96, 5.50],
        "G":     [31, 23, 28],
        "GS":    [31, 23, 28],
        "IP":    [192.0, 133.0, 158.0],
        "SO":    [228, 170, 90],
        "BB":    [35, 32, 60],
        "WHIP":  [0.92, 0.95, 1.45],
        "CG":    [1, 0, 0],
        "ShO":   [0, 0, 0],
        "SV":    [0, 0, 0],
        "FIP":   [2.49, 2.44, 4.80],
        "xFIP":  [2.93, 2.83, 4.50],
        "K-BB%": [25.5, 23.8, 5.1],
        "ERA-":  [60, 49, 130],
        "FIP-":  [62, 60, 110],
        "WAR":   [5.5, 4.3, 0.5],
        "xERA":  [2.85, 2.20, 5.20],
        "xwOBA": [0.255, 0.240, 0.350],
        "Stuff+":     [115, 130, 92],
        "Location+":  [105, 102, 96],
        "Pitching+":  [110, 117, 95],
        "Barrel%":    [5.5, 4.0, 9.0],
        "HardHit%":   [33.0, 30.0, 40.0],
        "RS/9":       [4.8, 5.2, 3.9],
    })


@pytest.fixture
def fake_bref_df():
    return pd.DataFrame({
        "Name":  ["Tarik Skubal", "Paul Skenes", "Bad Pitcher"],
        "Tm":    ["DET", "PIT", "OAK"],
        "W":     [18, 11, 4],
        "L":     [4, 3, 13],
        "bWAR":  [6.0, 4.5, 0.3],
        "HLD":   [0, 0, 0],
    })


@pytest.fixture
def fake_standings_df():
    """pybaseball.standings returns a list of DataFrames (one per division)."""
    return [
        pd.DataFrame({"Tm": ["DET", "CLE"], "W": [88, 92], "L": [74, 70]}),
        pd.DataFrame({"Tm": ["PIT", "MIL"], "W": [76, 93], "L": [86, 69]}),
        pd.DataFrame({"Tm": ["OAK", "HOU"], "W": [69, 88], "L": [93, 73]}),
    ]


@pytest.fixture
def fake_lahman_awards():
    return pd.DataFrame({
        "yearID":     [2024, 2024, 2024, 2024, 2023, 2023],
        "awardID":    ["Cy Young Award"] * 4 + ["Cy Young Award"] * 2,
        "lgID":       ["AL", "AL", "NL", "NL", "AL", "NL"],
        "playerID":   ["skubata01", "ragansh01", "skenepa01", "salech01", "coleger01", "snellbl01"],
        "pointsWon":  [210, 95, 90, 130, 200, 195],
        "pointsMax":  [210, 210, 210, 210, 210, 210],
        "votesFirst": [30, 0, 0, 14, 28, 26],
    })


@pytest.fixture
def fake_player_id_lookup():
    """Maps Lahman playerID -> Name for the players in fake_lahman_awards."""
    return pd.DataFrame({
        "key_bbref": ["skubata01", "ragansh01", "skenepa01", "salech01", "coleger01", "snellbl01"],
        "name_first": ["Tarik", "Hunter", "Paul", "Chris", "Gerrit", "Blake"],
        "name_last":  ["Skubal", "Roberts", "Skenes", "Sale", "Cole", "Snell"],
    })
