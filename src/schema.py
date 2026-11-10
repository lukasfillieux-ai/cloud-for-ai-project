"""Hoe de data eruit hoort te zien (Pandera). Breekt luid bij kapotte data."""

import pandera.pandas as pa

GEBOUWEN = pa.DataFrameSchema(
    {
        "site_id": pa.Column(int, pa.Check.ge(0)),
        "building_id": pa.Column(int, pa.Check.ge(0)),
        "primary_use": pa.Column(str),
        "square_feet": pa.Column(float, pa.Check.gt(0)),
        "year_built": pa.Column(float, nullable=True),
        "floor_count": pa.Column(float, nullable=True),
    },
    unique=["building_id"],
    coerce=True,
)

VERBRUIK = pa.DataFrameSchema(
    {
        "building_id": pa.Column(int, pa.Check.ge(0)),
        "meter": pa.Column(int, pa.Check.isin([0, 1, 2, 3])),
        "timestamp": pa.Column(pa.DateTime),
        # een verbruik kan nooit negatief zijn
        "meter_reading": pa.Column(float, pa.Check.ge(0)),
    },
    unique=["building_id", "meter", "timestamp"],
    coerce=True,
)

WEER = pa.DataFrameSchema(
    {
        "site_id": pa.Column(int, pa.Check.ge(0)),
        "timestamp": pa.Column(pa.DateTime),
        "air_temperature": pa.Column(float, pa.Check.in_range(-60, 60), nullable=True),
        "dew_temperature": pa.Column(float, pa.Check.in_range(-60, 60), nullable=True),
        "wind_speed": pa.Column(float, pa.Check.ge(0), nullable=True),
        "cloud_coverage": pa.Column(float, nullable=True),
        "precip_depth_1_hr": pa.Column(float, nullable=True),
    },
    unique=["site_id", "timestamp"],
    coerce=True,
)

VAKANTIES = pa.DataFrameSchema(
    {
        "site_id": pa.Column(int, pa.Check.ge(0)),
        "datum": pa.Column(pa.DateTime),
    },
    coerce=True,
)
