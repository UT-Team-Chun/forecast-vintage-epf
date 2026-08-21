"""JEPX area code enumeration."""

from __future__ import annotations

from enum import StrEnum


class AreaCode(StrEnum):
    """Japan electricity grid area codes used by JEPX/OCCTO.

    String values follow the lowercase short-name convention used in JEPX
    spot market CSV exports.

    Note: Okinawa is part of the national grid layout but is **not
    interconnected** with the main island grid, so it has no JEPX area
    price. Use :meth:`is_jepx_traded` to filter.
    """

    HOKKAIDO = "hokkaido"
    TOHOKU = "tohoku"
    TOKYO = "tokyo"
    CHUBU = "chubu"
    HOKURIKU = "hokuriku"
    KANSAI = "kansai"
    CHUGOKU = "chugoku"
    SHIKOKU = "shikoku"
    KYUSHU = "kyushu"
    OKINAWA = "okinawa"

    def is_jepx_traded(self) -> bool:
        """Return True if this area participates in the JEPX spot market."""
        return self is not AreaCode.OKINAWA

    @classmethod
    def jepx_traded_areas(cls) -> tuple[AreaCode, ...]:
        """Return the nine JEPX-traded areas in JEPX CSV column order."""
        return (
            cls.HOKKAIDO,
            cls.TOHOKU,
            cls.TOKYO,
            cls.CHUBU,
            cls.HOKURIKU,
            cls.KANSAI,
            cls.CHUGOKU,
            cls.SHIKOKU,
            cls.KYUSHU,
        )

    @property
    def japanese_name(self) -> str:
        """Return the Japanese kanji area name used in JEPX CSV column headers."""
        return _JP_NAMES[self]


_JP_NAMES: dict[AreaCode, str] = {
    AreaCode.HOKKAIDO: "北海道",
    AreaCode.TOHOKU: "東北",
    AreaCode.TOKYO: "東京",
    AreaCode.CHUBU: "中部",
    AreaCode.HOKURIKU: "北陸",
    AreaCode.KANSAI: "関西",
    AreaCode.CHUGOKU: "中国",
    AreaCode.SHIKOKU: "四国",
    AreaCode.KYUSHU: "九州",
    AreaCode.OKINAWA: "沖縄",
}
