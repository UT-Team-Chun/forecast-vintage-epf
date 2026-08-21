"""JEPX area price record."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from alg.models.epf.area import AreaCode  # noqa: TC001 - Pydantic runtime field


class PriceRecord(BaseModel):
    """A single half-hourly JEPX spot-market price observation.

    All timestamps are timezone-aware. ``price_yen_kwh`` represents the
    area price; ``system_price_yen_kwh`` is the nationwide system price
    when available.
    """

    model_config = ConfigDict(frozen=True)

    datetime_jst: datetime
    area: AreaCode
    price_yen_kwh: float = Field(ge=0.0)
    system_price_yen_kwh: float | None = Field(default=None, ge=0.0)
    volume_mwh: float | None = Field(default=None, ge=0.0)

    @field_validator("datetime_jst")
    @classmethod
    def _require_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "datetime_jst must be timezone-aware"
            raise ValueError(msg)
        return value
