"""Metrieken uit het scopingdocument: RMSLE, CV(RMSE), NMBE en de kost van onevenwicht."""

import numpy as np
import pandas as pd


def rmsle(y, p):
    p = np.clip(p, 0, None)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def cv_rmse(y, p):
    """RMSE gedeeld door het gemiddelde gemeten verbruik, in procent."""
    return float(100 * np.sqrt(np.mean((y - p) ** 2)) / np.mean(y))


def nmbe(y, p):
    """Som van (voorspeld - gemeten) gedeeld door het totaal gemeten, in procent.
    Positief = het model zit gemiddeld te hoog, negatief = te laag."""
    return float(100 * np.sum(p - y) / np.sum(y))


def optimaal_kwantiel(kost_fp, kost_fn):
    """Het kwantiel dat de verwachte kost minimaliseert: kost_fn / (kost_fn + kost_fp)."""
    return kost_fn / (kost_fn + kost_fp)


def pinball(y, p, q):
    """Pinball-verlies: de metriek die bij een kwantielvoorspelling past."""
    verschil = y - p
    return float(np.mean(np.maximum(q * verschil, (q - 1) * verschil)))


def kost_onevenwicht(y, p, kost_fp, kost_fn):
    """Euro: overschot (te veel voorspeld) kost kost_fp per kWh, tekort kost kost_fn per kWh."""
    overschot = np.clip(p - y, 0, None)
    tekort = np.clip(y - p, 0, None)
    return float(np.sum(overschot * kost_fp + tekort * kost_fn))


def kost_per_maand(timestamps, y, p, kost_fp, kost_fn):
    """Gemiddelde kost van onevenwicht in euro per maand."""
    overschot = np.clip(p - y, 0, None)
    tekort = np.clip(y - p, 0, None)
    kost = pd.Series(overschot * kost_fp + tekort * kost_fn)
    maand = pd.Series(pd.to_datetime(timestamps).to_numpy()).dt.to_period("M")
    return float(kost.groupby(maand).sum().mean())
