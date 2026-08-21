"""Calendar feature record (one timestamp = one row of cyclical encodings)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CalendarFeatures(BaseModel):
    """Cyclical + boolean calendar features derived from a JST timestamp.

    Sine/cosine encodings use the natural period of each field (24 for hour,
    7 for day-of-week, 12 for month). Holiday detection is delegated to
    ``alg.core.epf.features.calendar``.
    """

    model_config = ConfigDict(frozen=True)

    datetime_jst: datetime
    hour: int = Field(ge=0, le=23)
    dow: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_holiday: bool
    is_weekend: bool
    sin_hour: float = Field(ge=-1.0, le=1.0)
    cos_hour: float = Field(ge=-1.0, le=1.0)
    sin_dow: float = Field(ge=-1.0, le=1.0)
    cos_dow: float = Field(ge=-1.0, le=1.0)
    sin_month: float = Field(ge=-1.0, le=1.0)
    cos_month: float = Field(ge=-1.0, le=1.0)

    @field_validator("datetime_jst")
    @classmethod
    def _require_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "datetime_jst must be timezone-aware"
            raise ValueError(msg)
        return value
