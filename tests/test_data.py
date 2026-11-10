"""Klopt de data? Breekt het schema op kapotte data?"""

import pandas as pd
import pandera.pandas as pa
import pytest

from src.data import laad_gebouwen, verwijder_nulruns
from src.schema import VERBRUIK


def test_schema_accepteert_goede_data(tabel):
    VERBRUIK.validate(tabel[["building_id", "meter", "timestamp", "meter_reading"]])


def test_schema_weigert_negatief_verbruik(tabel):
    kapot = tabel[["building_id", "meter", "timestamp", "meter_reading"]].copy()
    kapot.loc[0, "meter_reading"] = -5.0
    with pytest.raises(pa.errors.SchemaError):
        VERBRUIK.validate(kapot)


def test_schema_weigert_dubbele_metingen(tabel):
    kapot = tabel[["building_id", "meter", "timestamp", "meter_reading"]].head(10)
    kapot = pd.concat([kapot, kapot.iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaError):
        VERBRUIK.validate(kapot)


def test_enkel_scholen_van_de_gekozen_sites(cfg):
    gebouwen = laad_gebouwen(cfg)
    assert set(gebouwen["primary_use"]) == {"Education"}
    assert 3 not in set(gebouwen["building_id"])  # het kantoor is eruit gefilterd


def test_lange_nulruns_weg_korte_blijven(tabel, cfg):
    gebouw0 = tabel[tabel["building_id"] == 0]
    assert len(gebouw0) == 150 * 24 - 100  # de 100 uur nullen zijn weg
    assert (gebouw0["meter_reading"] == 0).sum() == 5  # de 5 uur nullen blijven

    # de functie zelf: min_uren bepaalt de grens
    assert len(verwijder_nulruns(gebouw0, min_uren=3)) == len(gebouw0) - 5
