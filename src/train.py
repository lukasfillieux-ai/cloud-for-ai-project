"""Pijplijn bouwen, trainen, evalueren, wegschrijven. Draai met: python -m src.train"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import ROOT, laad_config
from src.data import laad_data, laad_vakanties
from src.doel import naar_log, terug_naar_kwh
from src.features import alle_features, bouw_featuretabel, numerieke_features
from src.metrics import (
    cv_rmse,
    kost_per_maand,
    nmbe,
    optimaal_kwantiel,
    pinball,
    rmsle,
)

MODEL_PAD = ROOT / "models" / "model.joblib"
METRICS_PAD = ROOT / "models" / "metrics.json"


def bouw_pijplijn(cfg):
    """Alles wat iets leert uit de data zit hier in: de encoder en het model.

    We voorspellen het kwantiel q = kost_fn / (kost_fn + kost_fp) in plaats van het gemiddelde.
    We trainen op log1p(verbruik): een kwantiel verandert niet door een stijgende
    transformatie, en zo is de fout relatief, zoals RMSLE.
    """
    k, m = cfg["kosten"], cfg["model"]
    q = optimaal_kwantiel(k["fp_per_kwh"], k["fn_per_kwh"])
    cat = cfg["features"]["categorisch"]
    num = numerieke_features(cfg)

    voorbewerking = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                cat,
            ),
            ("num", "passthrough", num),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_iter=m["max_iter"],
        learning_rate=m["learning_rate"],
        max_leaf_nodes=m["max_leaf_nodes"],
        min_samples_leaf=m["min_samples_leaf"],
        categorical_features=[True] * len(cat) + [False] * len(num),
        random_state=cfg["seed"],
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("voor", voorbewerking), ("model", model)]),
        func=naar_log,
        inverse_func=terug_naar_kwh,
    )


def splits(tabel, cfg):
    """Tijdsgebonden split: alles voor `testvanaf` is train, de rest is test."""
    doel = cfg["doel"]["kolom"]
    tabel = tabel[tabel[doel].notna()]
    grens = pd.Timestamp(cfg["split"]["testvanaf"])
    return tabel[tabel["timestamp"] < grens], tabel[tabel["timestamp"] >= grens]


def evalueer(model, test, cfg):
    """Model tegenover de naïeve baseline (zelfde uur vorige week), op dezelfde rijen."""
    doel = cfg["doel"]["kolom"]
    baseline_kolom = f"lag_{cfg['features']['baseline_lag_uren']}"
    kosten = cfg["kosten"]
    q = optimaal_kwantiel(kosten["fp_per_kwh"], kosten["fn_per_kwh"])

    t = test.dropna(subset=[baseline_kolom])
    y = t[doel].to_numpy()
    voorspellingen = {
        "model": model.predict(t[alle_features(cfg)]),
        "baseline": t[baseline_kolom].to_numpy(),
    }
    resultaat = {}
    for naam, p in voorspellingen.items():
        resultaat[naam] = {
            "rmsle": rmsle(y, p),
            "cv_rmse_pct": cv_rmse(y, p),
            "nmbe_pct": nmbe(y, p),
            "pinball": pinball(y, p, q),
            "dekking": float(np.mean(y <= p)),
            "kost_per_maand_eur": kost_per_maand(
                t["timestamp"], y, p, kosten["fp_per_kwh"], kosten["fn_per_kwh"]
            ),
        }
    resultaat["kostratio"] = (
        resultaat["model"]["kost_per_maand_eur"] / resultaat["baseline"]["kost_per_maand_eur"]
    )
    return resultaat


def toon(resultaat, q):
    print(f"\nKwantiel van het model: {q:.2f}  (dekking hoort daar dichtbij te liggen)")
    print(f"{'':10}{'RMSLE':>8}{'CV(RMSE)%':>11}{'NMBE%':>8}{'dekking':>9}{'€/maand':>12}")
    for naam in ("model", "baseline"):
        r = resultaat[naam]
        print(
            f"{naam:10}{r['rmsle']:8.3f}{r['cv_rmse_pct']:11.1f}{r['nmbe_pct']:8.1f}"
            f"{r['dekking']:9.2f}{r['kost_per_maand_eur']:12,.0f}"
        )
    print(f"Kostratio model/baseline: {resultaat['kostratio']:.2f}")


def main():
    cfg = laad_config()
    tabel = laad_data(cfg)
    features = bouw_featuretabel(tabel, cfg, laad_vakanties(cfg))
    train, test = splits(features, cfg)
    print(f"Train: {len(train):,} rijen  |  Test: {len(test):,} rijen")

    model = bouw_pijplijn(cfg)
    model.fit(train[alle_features(cfg)], train[cfg["doel"]["kolom"]])

    resultaat = evalueer(model, test, cfg)
    k = cfg["kosten"]
    toon(resultaat, optimaal_kwantiel(k["fp_per_kwh"], k["fn_per_kwh"]))

    grens = cfg["kwaliteit"]["max_kostratio"]
    if resultaat["kostratio"] > grens:
        raise SystemExit(
            f"Model niet weggeschreven: kostratio {resultaat['kostratio']:.2f} > {grens}"
        )

    MODEL_PAD.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PAD)
    METRICS_PAD.write_text(json.dumps(resultaat, indent=2), encoding="utf-8")
    print(f"\nModel weggeschreven naar {MODEL_PAD}")


if __name__ == "__main__":
    main()
