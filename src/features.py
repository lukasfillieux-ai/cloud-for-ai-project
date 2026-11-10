"""Features die uit de tijd en het recente verbruik komen.

Regel: wat we kennen op het voorspelmoment mag erin, de toekomst niet. Elke lag is minstens
`horizon_uren` oud, dus een voorspelling voor de komende 24 uur gebruikt enkel verbruik dat
op dat moment al gemeten is. Hier wordt niets geleerd uit de data, dus dit hoort niet in de
Pipeline.
"""

import pandas as pd

KALENDER = ["uur", "weekdag", "maand", "is_weekend", "is_vakantie"]


def numerieke_features(cfg):
    f = cfg["features"]
    lags = [f"lag_{k}" for k in f["lags_uren"]]
    rollings = [f"rol_{w}" for w in f["rolling_vensters_uren"]]
    return KALENDER + f["weer"] + f["gebouw"] + lags + rollings


def alle_features(cfg):
    return cfg["features"]["categorisch"] + numerieke_features(cfg)


def kalender_features(df, vakanties=None):
    ts = df["timestamp"]
    df["uur"] = ts.dt.hour
    df["weekdag"] = ts.dt.dayofweek
    df["maand"] = ts.dt.month
    df["is_weekend"] = (df["weekdag"] >= 5).astype(int)
    if vakanties is None:
        df["is_vakantie"] = 0
    else:
        vrij = pd.MultiIndex.from_frame(vakanties[["site_id", "datum"]])
        dag = pd.MultiIndex.from_arrays([df["site_id"], ts.dt.normalize()])
        df["is_vakantie"] = dag.isin(vrij).astype(int)
    return df


def historiek_features(df, cfg):
    """Lags en voortschrijdende gemiddelden per gebouw, op een volledig uurlijks raster."""
    f = cfg["features"]
    doel = cfg["doel"]["kolom"]
    horizon = cfg["horizon_uren"]

    breed = df.pivot(index="timestamp", columns="building_id", values=doel).sort_index()
    uren = pd.date_range(breed.index.min(), breed.index.max(), freq="h")
    breed = breed.reindex(uren)  # ontbrekende uren worden NaN, zodat een shift echt uren is

    matrices = {f"lag_{k}": breed.shift(k) for k in f["lags_uren"]}
    bekend = breed.shift(horizon)  # wat op het voorspelmoment bekend is
    for w in f["rolling_vensters_uren"]:
        matrices[f"rol_{w}"] = bekend.rolling(w, min_periods=max(1, w // 2)).mean()

    for naam, matrix in matrices.items():
        lang = matrix.rename_axis(index="timestamp", columns="building_id").reset_index()
        lang = lang.melt(id_vars="timestamp", var_name="building_id", value_name=naam)
        df = df.merge(lang, on=["timestamp", "building_id"], how="left")
    return df


def bouw_featuretabel(df, cfg, vakanties=None):
    df = kalender_features(df.copy(), vakanties)
    return historiek_features(df, cfg)
