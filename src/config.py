"""Leest de config en geeft paden terug."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STANDAARD = ROOT / "configs" / "default.yaml"


def maak_pad(pad):
    """Relatieve paden uit de config zijn relatief aan de projectmap."""
    pad = Path(pad)
    return pad if pad.is_absolute() else ROOT / pad


def laad_config(pad=None):
    with open(pad or STANDAARD, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    controleer_config(cfg)
    return cfg


def controleer_config(cfg):
    horizon = cfg["horizon_uren"]
    lags = cfg["features"]["lags_uren"]
    if min(lags) < horizon:
        raise ValueError(
            f"Kleinste lag ({min(lags)} u) is korter dan de horizon ({horizon} u): "
            "het model zou de toekomst zien."
        )
    if cfg["features"]["baseline_lag_uren"] not in lags:
        raise ValueError("baseline_lag_uren moet een van de lags_uren zijn.")
