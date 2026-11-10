"""Nagemaakte data in het ASHRAE-formaat, zodat de tests draaien zonder de echte bestanden."""

import numpy as np
import pandas as pd
import pytest

from src.config import laad_config
from src.data import laad_data
from src.features import alle_features, bouw_featuretabel
from src.train import bouw_pijplijn, splits

UREN = pd.date_range("2016-01-01", periods=150 * 24, freq="h")


def _maak_csvs(map_):
    rng = np.random.default_rng(0)
    gebouwen = pd.DataFrame(
        {
            "site_id": [4, 4, 5, 4],
            "building_id": [0, 1, 2, 3],
            "primary_use": ["Education", "Education", "Education", "Office"],
            "square_feet": [50000.0, 80000.0, 30000.0, 20000.0],
            "year_built": [1990.0, np.nan, 1975.0, 2005.0],
            "floor_count": [2.0, 3.0, np.nan, 5.0],
        }
    )
    dag = UREN.dayofyear.to_numpy()
    uur = UREN.hour.to_numpy()
    temp = 10 + 12 * np.sin(2 * np.pi * (dag - 110) / 365) + 4 * np.sin(2 * np.pi * (uur - 9) / 24)

    weer = []
    for site in (4, 5):
        weer.append(
            pd.DataFrame(
                {
                    "site_id": site,
                    "timestamp": UREN,
                    "air_temperature": temp + rng.normal(0, 1, len(UREN)),
                    "dew_temperature": temp - 3,
                    "wind_speed": rng.uniform(0, 8, len(UREN)),
                    "cloud_coverage": rng.integers(0, 9, len(UREN)).astype(float),
                    "precip_depth_1_hr": rng.uniform(0, 2, len(UREN)),
                }
            )
        )
    weer = pd.concat(weer, ignore_index=True)

    schooluur = ((UREN.dayofweek < 5) & (UREN.hour >= 7) & (UREN.hour < 17)).astype(float)
    verbruik = []
    for _, g in gebouwen.iterrows():
        koude = np.clip(15 - temp, 0, None)
        waarde = g["square_feet"] / 100 * (0.2 + 0.8 * schooluur) * (1 + 0.06 * koude)
        waarde = waarde * rng.lognormal(0, 0.1, len(UREN))
        df = pd.DataFrame(
            {
                "building_id": int(g["building_id"]),
                "meter": 0,
                "timestamp": UREN,
                "meter_reading": waarde,
            }
        )
        if g["building_id"] == 0:
            df.loc[1000:1099, "meter_reading"] = 0.0  # defecte meter: 100 uur nullen
            df.loc[2000:2004, "meter_reading"] = 0.0  # korte reeks van 5 uur: blijft staan
        verbruik.append(df)
    verbruik = pd.concat(verbruik, ignore_index=True)

    gebouwen.to_csv(map_ / "building_metadata.csv", index=False)
    weer.to_csv(map_ / "weather_train.csv", index=False)
    verbruik.to_csv(map_ / "train.csv", index=False)


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    map_ = tmp_path_factory.mktemp("data")
    _maak_csvs(map_)
    cfg = laad_config()
    cfg["data"].update(
        verbruik_pad=str(map_ / "train.csv"),
        gebouwen_pad=str(map_ / "building_metadata.csv"),
        weer_pad=str(map_ / "weather_train.csv"),
        chunksize=5000,
    )
    cfg["split"]["testvanaf"] = "2016-05-01"
    cfg["model"].update(max_iter=80, min_samples_leaf=20)
    return cfg


@pytest.fixture(scope="session")
def tabel(cfg):
    return laad_data(cfg)


@pytest.fixture(scope="session")
def features(cfg, tabel):
    return bouw_featuretabel(tabel, cfg)


@pytest.fixture(scope="session")
def train_test(cfg, features):
    return splits(features, cfg)


@pytest.fixture(scope="session")
def model(cfg, train_test):
    train, _ = train_test
    return bouw_pijplijn(cfg).fit(train[alle_features(cfg)], train[cfg["doel"]["kolom"]])
