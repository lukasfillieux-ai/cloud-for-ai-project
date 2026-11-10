"""De dingen die je wél kan testen aan een model."""

import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT
from src.features import alle_features
from src.metrics import optimaal_kwantiel
from src.predict import voorspel_dag
from src.train import bouw_pijplijn, evalueer


def test_voorspellingen_hebben_de_juiste_vorm_en_zijn_geldig(model, train_test, cfg):
    _, test = train_test
    p = model.predict(test[alle_features(cfg)])
    assert p.shape == (len(test),)
    assert not np.isnan(p).any()
    assert (p >= 0).all()  # een verbruik is nooit negatief


def test_zelfde_seed_geeft_zelfde_model(model, train_test, cfg):
    train, test = train_test
    tweede = bouw_pijplijn(cfg).fit(train[alle_features(cfg)], train[cfg["doel"]["kolom"]])
    kolommen = alle_features(cfg)
    np.testing.assert_allclose(model.predict(test[kolommen]), tweede.predict(test[kolommen]))


def test_dekking_ligt_dicht_bij_het_gekozen_kwantiel(model, train_test, cfg):
    """Bij q = 0,75 hoort ongeveer 75% van de metingen onder de voorspelling te liggen."""
    _, test = train_test
    p = model.predict(test[alle_features(cfg)])
    dekking = np.mean(test[cfg["doel"]["kolom"]].to_numpy() <= p)
    q = optimaal_kwantiel(cfg["kosten"]["fp_per_kwh"], cfg["kosten"]["fn_per_kwh"])
    assert abs(dekking - q) < 0.1


def test_koud_weer_geeft_meer_verbruik_dan_warm_weer(model, train_test, cfg):
    _, test = train_test
    koud, warm = test.copy(), test.copy()
    koud["air_temperature"], warm["air_temperature"] = -5.0, 25.0
    kolommen = alle_features(cfg)
    assert model.predict(koud[kolommen]).mean() > model.predict(warm[kolommen]).mean()


def test_model_is_goedkoper_dan_de_naieve_baseline(model, train_test, cfg):
    _, test = train_test
    resultaat = evalueer(model, test, cfg)
    assert resultaat["kostratio"] <= cfg["kwaliteit"]["max_kostratio"]


def test_opslaan_en_laden_geeft_dezelfde_voorspellingen(model, train_test, cfg, tmp_path):
    _, test = train_test
    joblib.dump(model, tmp_path / "m.joblib")
    geladen = joblib.load(tmp_path / "m.joblib")
    kolommen = alle_features(cfg)
    np.testing.assert_allclose(model.predict(test[kolommen]), geladen.predict(test[kolommen]))


def test_voorspel_dag_geeft_24_uur_per_gebouw(model, cfg, tabel):
    uitvoer = voorspel_dag(model, cfg, "2016-05-10", tabel=tabel)
    assert len(uitvoer) == 3 * 24  # drie scholen, 24 uur
    assert uitvoer["voorspelling_kwh"].notna().all()
    assert uitvoer["timestamp"].dt.date.nunique() == 1


def test_voorspel_dag_negeert_het_verbruik_van_die_dag(model, cfg, tabel):
    """Wat er op de doeldag zelf gemeten werd mag de voorspelling niet beïnvloeden."""
    normaal = voorspel_dag(model, cfg, "2016-05-10", tabel=tabel)
    gewijzigd = tabel.copy()
    gewijzigd.loc[gewijzigd["timestamp"] >= pd.Timestamp("2016-05-10"), "meter_reading"] *= 100
    na_wijziging = voorspel_dag(model, cfg, "2016-05-10", tabel=gewijzigd)
    pd.testing.assert_frame_equal(normaal, na_wijziging)


def test_model_laadt_in_een_vers_proces(model, tmp_path):
    """Zoals `python -m src.predict` het doet: een ander proces dan het trainen."""
    joblib.dump(model, tmp_path / "m.joblib")
    code = "import joblib, sys; joblib.load(sys.argv[1])"
    klaar = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "m.joblib")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert klaar.returncode == 0, klaar.stderr
