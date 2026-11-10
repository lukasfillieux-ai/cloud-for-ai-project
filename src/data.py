"""Inlezen, valideren en opschonen. Alle paden komen uit de config."""

import pandas as pd

from src.config import maak_pad
from src.schema import GEBOUWEN, VAKANTIES, VERBRUIK, WEER


def laad_gebouwen(cfg):
    d = cfg["data"]
    meta = pd.read_csv(maak_pad(d["gebouwen_pad"]))
    meta = meta[meta["primary_use"] == d["primary_use"]]
    if d["site_ids"] is not None:
        meta = meta[meta["site_id"].isin(d["site_ids"])]
    return GEBOUWEN.validate(meta.reset_index(drop=True))


def laad_verbruik(cfg, building_ids):
    """Leest het grote bestand in stukken, en houdt enkel onze gebouwen en meter over."""
    d = cfg["data"]
    ids = set(building_ids)
    stukken = []
    for chunk in pd.read_csv(
        maak_pad(d["verbruik_pad"]), parse_dates=["timestamp"], chunksize=d["chunksize"]
    ):
        stukken.append(chunk[(chunk["meter"] == d["meter"]) & chunk["building_id"].isin(ids)])
    verbruik = pd.concat(stukken, ignore_index=True)
    return VERBRUIK.validate(verbruik)


def laad_weer(cfg):
    weer = pd.read_csv(maak_pad(cfg["data"]["weer_pad"]), parse_dates=["timestamp"])
    kolommen = ["site_id", "timestamp", *cfg["features"]["weer"]]
    kolommen = list(dict.fromkeys(kolommen))
    return WEER.validate(weer[kolommen])


def laad_vakanties(cfg):
    """Optioneel: de schoolvakantiekalender. Geen bestand in de config = geen vakanties."""
    pad = cfg["data"]["vakantie_pad"]
    if pad is None:
        return None
    return VAKANTIES.validate(pd.read_csv(maak_pad(pad), parse_dates=["datum"]))


def verwijder_nulruns(df, min_uren, kolom="meter_reading"):
    """Gooit reeksen van minstens `min_uren` opeenvolgende nullen weg (defecte meter)."""
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    nul = df[kolom].eq(0)
    # een nieuwe reeks begint bij een wissel nul/niet-nul of bij een nieuw gebouw
    wissel = nul.ne(nul.shift()) | df["building_id"].ne(df["building_id"].shift())
    lengte = nul.groupby(wissel.cumsum()).transform("size")
    return df[~(nul & (lengte >= min_uren))].reset_index(drop=True)


def laad_data(cfg):
    """Verbruik + gebouw + weer, opgeschoond. Eén rij per gebouw per uur."""
    gebouwen = laad_gebouwen(cfg)
    verbruik = laad_verbruik(cfg, gebouwen["building_id"])
    verbruik = verwijder_nulruns(verbruik, cfg["data"]["min_nulrun_uren"], cfg["doel"]["kolom"])
    tabel = verbruik.merge(gebouwen, on="building_id", how="left")
    tabel = tabel.merge(laad_weer(cfg), on=["site_id", "timestamp"], how="left")
    return tabel
