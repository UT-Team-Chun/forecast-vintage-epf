"""Point + quantile forecast record."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from alg.models.epf.area import AreaCode  # noqa: TC001 - Pydantic runtime field


class Forecast(BaseModel):
    """A single forecast emission combining a point estimate with three quantiles.

    The point estimate ``mean`` is independent of the quantiles, but ``q10``,
    ``q50`` and ``q90`` are constrained to be monotonically increasing.
    """

    model_config = ConfigDict(frozen=True)

    datetime_jst: datetime
    area: AreaCode
    mean: float
    q10: float
    q50: float
    q90: float

    @field_validator("datetime_jst")
    @classmethod
    def _require_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "datetime_jst must be timezone-aware"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _check_quantile_order(self) -> Self:
        if not (self.q10 <= self.q50 <= self.q90):
            msg = (
                f"Quantiles must be non-decreasing: q10={self.q10}, q50={self.q50}, q90={self.q90}"
            )
            raise ValueError(msg)
        return self
