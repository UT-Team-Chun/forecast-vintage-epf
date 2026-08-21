"""Calendar feature engineering for JEPX-style timestamps.

Produces both:
- A polars DataFrame ready for direct concatenation with numeric features.
- Per-timestamp ``CalendarFeatures`` Pydantic records for unit tests and
  documentation of the feature schema.
"""

from __future__ import annotations

import math
from datetime import datetime

import jpholiday
import polars as pl

from alg.models.epf.calendar import CalendarFeatures

_DEFAULT_DT_COL = "datetime_jst"


def _sin_cos(value: float, period: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(value) / float(period)
    return math.sin(angle), math.cos(angle)


def features_for_timestamp(ts: datetime) -> CalendarFeatures:
    """Compute calendar features for a single tz-aware timestamp."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        msg = "Timestamp must be timezone-aware"
        raise ValueError(msg)
    hour = ts.hour
    dow = ts.weekday()
    month = ts.month
    is_holiday = jpholiday.is_holiday(ts.date())
    is_weekend = dow >= 5  # noqa: PLR2004 - 5=Sat, 6=Sun is unambiguous
    sin_h, cos_h = _sin_cos(hour, 24)
    sin_d, cos_d = _sin_cos(dow, 7)
    sin_m, cos_m = _sin_cos(month - 1, 12)
    return CalendarFeatures(
        datetime_jst=ts,
        hour=hour,
        dow=dow,
        month=month,
        is_holiday=is_holiday,
        is_weekend=is_weekend,
        sin_hour=sin_h,
        cos_hour=cos_h,
        sin_dow=sin_d,
        cos_dow=cos_d,
        sin_month=sin_m,
        cos_month=cos_m,
    )


def make_calendar_features(
    df: pl.DataFrame,
    *,
    datetime_col: str = _DEFAULT_DT_COL,
) -> pl.DataFrame:
    """Append calendar features to a polars DataFrame.

    The input DataFrame must contain a tz-aware Datetime column named
    ``datetime_col``. The output adds: ``hour``, ``dow``, ``month``,
    ``is_holiday``, ``is_weekend``, ``sin_hour``/``cos_hour``,
    ``sin_dow``/``cos_dow``, ``sin_month``/``cos_month``.
    """
    if datetime_col not in df.columns:
        msg = f"DataFrame is missing required column '{datetime_col}'"
        raise ValueError(msg)
    dtype = df.schema[datetime_col]
    if not isinstance(dtype, pl.Datetime):
        msg = f"Column '{datetime_col}' must be polars Datetime, got {dtype}"
        raise TypeError(msg)
    if dtype.time_zone is None:
        msg = f"Column '{datetime_col}' must be timezone-aware"
        raise ValueError(msg)

    hours = df[datetime_col].dt.hour()
    dows = df[datetime_col].dt.weekday() - 1  # polars: Mon=1..Sun=7 -> 0..6
    months = df[datetime_col].dt.month()

    sin_hour = hours.map_elements(
        lambda h: math.sin(2.0 * math.pi * float(h) / 24.0),
        return_dtype=pl.Float64,
    )
    cos_hour = hours.map_elements(
        lambda h: math.cos(2.0 * math.pi * float(h) / 24.0),
        return_dtype=pl.Float64,
    )
    sin_dow = dows.map_elements(
        lambda d: math.sin(2.0 * math.pi * float(d) / 7.0),
        return_dtype=pl.Float64,
    )
    cos_dow = dows.map_elements(
        lambda d: math.cos(2.0 * math.pi * float(d) / 7.0),
        return_dtype=pl.Float64,
    )
    sin_month = months.map_elements(
        lambda m: math.sin(2.0 * math.pi * (float(m) - 1.0) / 12.0),
        return_dtype=pl.Float64,
    )
    cos_month = months.map_elements(
        lambda m: math.cos(2.0 * math.pi * (float(m) - 1.0) / 12.0),
        return_dtype=pl.Float64,
    )

    holiday_flags = [jpholiday.is_holiday(d) for d in df[datetime_col].dt.date().to_list()]
    weekend_flags = [(d >= 5) for d in dows.to_list()]  # noqa: PLR2004

    return df.with_columns(
        hour=hours.cast(pl.Int8),
        dow=dows.cast(pl.Int8),
        month=months.cast(pl.Int8),
        is_holiday=pl.Series(holiday_flags, dtype=pl.Boolean),
        is_weekend=pl.Series(weekend_flags, dtype=pl.Boolean),
        sin_hour=sin_hour,
        cos_hour=cos_hour,
        sin_dow=sin_dow,
        cos_dow=cos_dow,
        sin_month=sin_month,
        cos_month=cos_month,
    )
