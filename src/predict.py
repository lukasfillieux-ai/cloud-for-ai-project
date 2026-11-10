"""Model laden en een dag voorspellen. Draai met: python -m src.predict 2016-12-15

Gebruikt enkel verbruik van vóór het begin van die dag, plus het weer van die dag
(in de dataset gemeten, in productie een weersvoorspelling).
"""

import sys

import joblib
import numpy as np
import pandas as pd

from src.config import ROOT, laad_config
from src.data import laad_data, laad_gebouwen, laad_vakanties, laad_weer
from src.features import alle_features, bouw_featuretabel
from src.train import MODEL_PAD


def voorspel_dag(model, cfg, dag, tabel=None):
    """Voorspelt per gebouw het verbruik (kWh) voor de 24 uur vanaf `dag` 00:00."""
    doel = cfg["doel"]["kolom"]
    start = pd.Timestamp(dag)
    uren = pd.date_range(start, periods=cfg["horizon_uren"], freq="h")

    if tabel is None:
        tabel = laad_data(cfg)
    historiek = tabel[tabel["timestamp"] < start]

    # een leeg raster van gebouwen x uren van de doeldag, met het weer van die dag
    gebouwen = laad_gebouwen(cfg)
    gebouwen = gebouwen[gebouwen["building_id"].isin(historiek["building_id"].unique())]
    raster = gebouwen.merge(pd.DataFrame({"timestamp": uren}), how="cross")
    raster = raster.merge(laad_weer(cfg), on=["site_id", "timestamp"], how="left")
    raster[doel] = np.nan

    alles = pd.concat([historiek, raster], ignore_index=True)
    features = bouw_featuretabel(alles, cfg, laad_vakanties(cfg))
    dagrijen = features[features["timestamp"].isin(uren)].copy()
    dagrijen["voorspelling_kwh"] = model.predict(dagrijen[alle_features(cfg)])
    return dagrijen[["building_id", "timestamp", "voorspelling_kwh"]].reset_index(drop=True)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Gebruik: python -m src.predict JJJJ-MM-DD")
    dag = sys.argv[1]
    cfg = laad_config()
    model = joblib.load(MODEL_PAD)
    resultaat = voorspel_dag(model, cfg, dag)

    uitvoer = ROOT / "outputs" / f"voorspelling_{dag}.csv"
    uitvoer.parent.mkdir(exist_ok=True)
    resultaat.to_csv(uitvoer, index=False)
    print(f"{len(resultaat)} voorspellingen weggeschreven naar {uitvoer}")


if __name__ == "__main__":
    main()
