"""Omrekening van het doel. Staat in een eigen module, zodat joblib het model ook kan laden
vanuit een ander script dan het script dat het getraind heeft (src.predict versus src.train)."""

import numpy as np


def naar_log(y):
    return np.log1p(y)


def terug_naar_kwh(z):
    """Omgekeerde van log1p, nooit onder nul (een verbruik is niet negatief)."""
    return np.clip(np.expm1(z), 0, None)
