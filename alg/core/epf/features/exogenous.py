"""Time-aligning exogenous numeric signals onto the JEPX target grid.

JEPX publishes spot prices on a 30-minute grid. Several upstream signals
(area demand, ERA5 weather) are published on a 1-hour grid. To use them
as exogenous features we align both onto the same timestamps by joining
on the floored hour of each JEPX timestamp: the two half-hour slots
``[H:00, H:30)`` and ``[H:30, H+1:00)`` inherit the same hourly value.

``attach_hourly_features`` is the generic helper. Thin wrappers
``attach_hourly_demand`` and ``attach_hourly_weather`` exist so callers
don't have to remember the column list of each source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_DT_COL = "datetime_jst"
DEFAULT_DEMAND_COL = "demand_mw"
DEFAULT_WEATHER_COLS: tuple[str, ...] = (
    "temperature_c",
    "cloud_cover_pct",
    "shortwave_radiation_wm2",
    "wind_speed_kmh",
)
DEFAULT_SATELLITE_COLS: tuple[str, ...] = (
    "sat_visible_mean",
    "sat_visible_std",
    "sat_ir_brightness_temp",
)


def attach_hourly_features(
    price_df: pl.DataFrame,
    features_df: pl.DataFrame,
    *,
    feature_cols: Iterable[str],
    datetime_col: str = DEFAULT_DT_COL,
) -> pl.DataFrame:
    """Return ``price_df`` with selected ``feature_cols`` joined by hour-floor.

    Both DataFrames must contain ``datetime_col`` as tz-aware Datetime.
    ``features_df`` is expected to have a row per hour; ``price_df`` may
    be at any sub-hour cadence. Missing hours produce nulls.
    """
    cols = tuple(feature_cols)
    if not cols:
        msg = "feature_cols must contain at least one column"
        raise ValueError(msg)
    if datetime_col not in price_df.columns:
        msg = f"price_df is missing required column '{datetime_col}'"
        raise ValueError(msg)
    if datetime_col not in features_df.columns:
        msg = f"features_df is missing required column '{datetime_col}'"
        raise ValueError(msg)
    for col in cols:
        if col not in features_df.columns:
            msg = f"features_df is missing required column '{col}'"
            raise ValueError(msg)
    price_dtype = price_df.schema[datetime_col]
    features_dtype = features_df.schema[datetime_col]
    if not isinstance(price_dtype, pl.Datetime) or not isinstance(features_dtype, pl.Datetime):
        msg = "Both DataFrames must use polars Datetime for the join column"
        raise TypeError(msg)
    if price_dtype.time_zone is None or features_dtype.time_zone is None:
        msg = "Both join columns must be tz-aware"
        raise ValueError(msg)

    compact = features_df.select(
        pl.col(datetime_col).alias("_features_hour"),
        *(pl.col(c) for c in cols),
    )
    return (
        price_df.with_columns(
            pl.col(datetime_col).dt.truncate("1h").alias("_features_hour"),
        )
        .join(compact, on="_features_hour", how="left")
        .drop("_features_hour")
    )


def attach_hourly_demand(
    price_df: pl.DataFrame,
    demand_df: pl.DataFrame,
    *,
    datetime_col: str = DEFAULT_DT_COL,
    demand_col: str = DEFAULT_DEMAND_COL,
) -> pl.DataFrame:
    """Backward-compatible wrapper that attaches a single demand column."""
    return attach_hourly_features(
        price_df,
        demand_df,
        feature_cols=(demand_col,),
        datetime_col=datetime_col,
    )


def attach_hourly_weather(
    price_df: pl.DataFrame,
    weather_df: pl.DataFrame,
    *,
    feature_cols: Iterable[str] = DEFAULT_WEATHER_COLS,
    datetime_col: str = DEFAULT_DT_COL,
) -> pl.DataFrame:
    """Attach Open-Meteo-style hourly weather features to ``price_df``."""
    return attach_hourly_features(
        price_df,
        weather_df,
        feature_cols=feature_cols,
        datetime_col=datetime_col,
    )


def attach_hourly_satellite(
    price_df: pl.DataFrame,
    satellite_df: pl.DataFrame,
    *,
    feature_cols: Iterable[str] = DEFAULT_SATELLITE_COLS,
    datetime_col: str = DEFAULT_DT_COL,
) -> pl.DataFrame:
    """Attach Himawari-style hourly region-aggregated satellite features.

    ``satellite_df`` is expected to be the output of either
    :func:`alg.core.epf.data.synthetic.generate_synthetic_satellite`
    or :func:`alg.core.epf.data.himawari_loader.load_himawari_satellite`.
    Half-hour JEPX slots inherit the same hourly satellite values via
    hour-floor join, mirroring the demand / weather paths.
    """
    return attach_hourly_features(
        price_df,
        satellite_df,
        feature_cols=feature_cols,
        datetime_col=datetime_col,
    )
