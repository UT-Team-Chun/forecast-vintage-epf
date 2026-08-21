"""Regime label and probability data contracts."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class RegimeLabel(IntEnum):
    """Three-class regime classification used throughout the EPF harness."""

    FLOOR = 0
    NORMAL = 1
    SPIKE = 2


class RegimeProbabilities(BaseModel):
    """Soft regime assignment as a probability vector.

    The three probabilities must each be in ``[0, 1]`` and sum to ``1`` within
    a small numerical tolerance.
    """

    p_floor: float = Field(ge=0.0, le=1.0)
    p_normal: float = Field(ge=0.0, le=1.0)
    p_spike: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> Self:
        total = self.p_floor + self.p_normal + self.p_spike
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            msg = f"Regime probabilities must sum to 1.0, got {total:.6f}"
            raise ValueError(msg)
        return self

    @classmethod
    def uniform(cls) -> RegimeProbabilities:
        """Return a (1/3, 1/3, 1/3) probability vector."""
        third = 1.0 / 3.0
        return cls(p_floor=third, p_normal=third, p_spike=1.0 - 2 * third)

    def as_tuple(self) -> tuple[float, float, float]:
        """Return ``(p_floor, p_normal, p_spike)``."""
        return self.p_floor, self.p_normal, self.p_spike
