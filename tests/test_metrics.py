"""De metrieken uit het scopingdocument, op getallen die je met de hand kan narekenen."""

import numpy as np
import pytest

from src.metrics import cv_rmse, kost_onevenwicht, nmbe, optimaal_kwantiel, rmsle


def test_perfecte_voorspelling_geeft_nul_fout():
    y = np.array([10.0, 20.0, 30.0])
    assert rmsle(y, y) == 0
    assert cv_rmse(y, y) == 0
    assert nmbe(y, y) == 0


def test_nmbe_teken_zegt_te_hoog_of_te_laag():
    y = np.array([100.0, 100.0])
    assert nmbe(y, y * 1.1) == pytest.approx(10.0)  # te hoog
    assert nmbe(y, y * 0.9) == pytest.approx(-10.0)  # te laag


def test_een_tekort_kost_drie_keer_zoveel_als_een_overschot():
    y, p_laag, p_hoog = np.array([100.0]), np.array([90.0]), np.array([110.0])
    tekort = kost_onevenwicht(y, p_laag, kost_fp=0.02, kost_fn=0.06)
    overschot = kost_onevenwicht(y, p_hoog, kost_fp=0.02, kost_fn=0.06)
    assert tekort == pytest.approx(3 * overschot)


def test_optimaal_kwantiel_is_75_procent():
    assert optimaal_kwantiel(0.02, 0.06) == pytest.approx(0.75)
