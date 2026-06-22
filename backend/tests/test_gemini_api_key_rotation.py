"""Gemini API key IST rotation."""

from datetime import datetime

from zoneinfo import ZoneInfo

from app.utils.gemini_api_key import IST, resolve_rotating_gemini_api_key

SAMPATH = "key-sampath"
RISHI = "key-rishi"
SYAM = "key-syam"


def _ist(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 6, 5, hour, minute, tzinfo=IST)


def test_sampath_slot_morning() -> None:
    key, slot = resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(9, 30)
    )
    assert key == SAMPATH
    assert slot == "Sampath"


def test_rishi_slot_afternoon() -> None:
    key, slot = resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(14, 0)
    )
    assert key == RISHI
    assert slot == "Rishi"


def test_syam_slot_evening() -> None:
    key, slot = resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(18, 45)
    )
    assert key == SYAM
    assert slot == "Syam"


def test_rishi_falls_back_to_sampath_when_empty() -> None:
    key, slot = resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=None, syam=SYAM, now=_ist(13, 0)
    )
    assert key == SAMPATH
    assert slot == "Rishi→Sampath"


def test_syam_falls_back_to_sampath_when_empty() -> None:
    key, slot = resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam="", now=_ist(17, 0)
    )
    assert key == SAMPATH
    assert slot == "Syam→Sampath"


def test_off_hours_use_sampath() -> None:
    key, slot = resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(22, 0)
    )
    assert key == SAMPATH
    assert slot == "Sampath (off-hours)"


def test_slot_boundaries() -> None:
    assert resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(8, 0)
    ) == (SAMPATH, "Sampath")
    assert resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(12, 0)
    ) == (RISHI, "Rishi")
    assert resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(16, 0)
    ) == (SYAM, "Syam")
    assert resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=_ist(21, 0)
    ) == (SAMPATH, "Sampath (off-hours)")


def test_legacy_google_api_key_as_sampath_fallback() -> None:
    key, slot = resolve_rotating_gemini_api_key(
        sampath=None, rishi=RISHI, syam=SYAM, legacy=SAMPATH, now=_ist(10, 0)
    )
    assert key == SAMPATH
    assert slot == "Sampath"


def test_uses_ist_not_utc() -> None:
    # 06:30 UTC = 12:00 IST → Rishi slot
    utc_noon_ist = datetime(2026, 6, 5, 6, 30, tzinfo=ZoneInfo("UTC"))
    key, slot = resolve_rotating_gemini_api_key(
        sampath=SAMPATH, rishi=RISHI, syam=SYAM, now=utc_noon_ist
    )
    assert key == RISHI
    assert slot == "Rishi"
