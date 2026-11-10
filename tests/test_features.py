"""Geen lekken uit de toekomst: dit is de test die het meeste telt voor een voorspelmodel."""

import pandas as pd
import pytest

from src.config import controleer_config
from src.features import bouw_featuretabel


def test_lag_24_is_het_verbruik_van_24_uur_eerder(features):
    g = features[features["building_id"] == 1].set_index("timestamp")
    t = pd.Timestamp("2016-03-10 14:00")
    assert g.loc[t, "lag_24"] == pytest.approx(g.loc[t - pd.Timedelta(hours=24), "meter_reading"])
    assert g.loc[t, "lag_168"] == pytest.approx(g.loc[t - pd.Timedelta(hours=168), "meter_reading"])


def test_features_zien_de_toekomst_niet(tabel, cfg, features):
    """Verander het verbruik vanaf T. Features voor tijdstippen vóór T + 24 u blijven gelijk."""
    grens = pd.Timestamp("2016-03-15")
    aangepast = tabel.copy()
    aangepast.loc[aangepast["timestamp"] >= grens, "meter_reading"] *= 10
    nieuw = bouw_featuretabel(aangepast, cfg)

    kolommen = ["lag_24", "lag_48", "lag_168", "rol_24", "rol_168"]
    veilig = features["timestamp"] < grens + pd.Timedelta(hours=cfg["horizon_uren"])
    pd.testing.assert_frame_equal(
        features.loc[veilig, kolommen].reset_index(drop=True),
        nieuw.loc[veilig, kolommen].reset_index(drop=True),
    )


def test_lag_korter_dan_horizon_wordt_geweigerd(cfg):
    slecht = {**cfg, "features": {**cfg["features"], "lags_uren": [1, 24, 168]}}
    with pytest.raises(ValueError):
        controleer_config(slecht)
