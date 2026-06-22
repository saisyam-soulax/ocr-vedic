"""Gemini API key helpers — pool of all configured keys for round-robin rotation.

Previously keys were rotated on a fixed IST time schedule (one key per 4-hour
window).  That left two keys idle while a single key exhausted its daily quota.

The new design exposes all configured keys as a pool.  Pages are distributed
round-robin across the pool, and any key that returns 429 RESOURCE_EXHAUSTED is
immediately skipped in favour of the next key.

The legacy ``resolve_rotating_gemini_api_key`` is kept for backward compat
(config.py's ``effective_google_api_key`` still uses it as the single-key
default shown in provider status checks).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Minutes from midnight in IST: 08:00, 12:00, 16:00, 21:00
_SLOT_SAMPATH_START = 8 * 60
_SLOT_RISHI_START = 12 * 60
_SLOT_SYAM_START = 16 * 60
_SLOT_SYAM_END = 21 * 60


def _normalize_key(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def resolve_rotating_gemini_api_key(
    *,
    sampath: str | None,
    rishi: str | None,
    syam: str | None,
    legacy: str | None = None,
    now: datetime | None = None,
) -> tuple[str | None, str]:
    """Pick the active Gemini API key from the IST schedule.

    Slots (inclusive start, exclusive end):
      Sampath — 08:00–12:00 IST
      Rishi   — 12:00–16:00 IST (falls back to Sampath when empty)
      Syam    — 16:00–21:00 IST (falls back to Sampath when empty)
      Off-hours (21:00–08:00) — Sampath

    ``legacy`` is the optional ``GOOGLE_API_KEY`` value used as Sampath when
  ``sampath`` is unset (backward compatibility).
    """
    sampath_key = _normalize_key(sampath) or _normalize_key(legacy)
    rishi_key = _normalize_key(rishi)
    syam_key = _normalize_key(syam)

    if now is None:
        ist_now = datetime.now(IST)
    elif now.tzinfo is None:
        ist_now = now.replace(tzinfo=IST)
    else:
        ist_now = now.astimezone(IST)

    minutes = ist_now.hour * 60 + ist_now.minute

    if _SLOT_SAMPATH_START <= minutes < _SLOT_RISHI_START:
        return sampath_key, "Sampath"
    if _SLOT_RISHI_START <= minutes < _SLOT_SYAM_START:
        if rishi_key:
            return rishi_key, "Rishi"
        return sampath_key, "Rishi→Sampath"
    if _SLOT_SYAM_START <= minutes < _SLOT_SYAM_END:
        if syam_key:
            return syam_key, "Syam"
        return sampath_key, "Syam→Sampath"
    return sampath_key, "Sampath (off-hours)"


def all_configured_keys(
    *,
    sampath: str | None,
    rishi: str | None,
    syam: str | None,
    legacy: str | None = None,
) -> list[tuple[str, str]]:
    """Return ALL non-empty keys as ``[(api_key, slot_name), ...]`` in canonical order.

    Canonical order: Sampath → Rishi → Syam.  ``legacy`` (bare GOOGLE_API_KEY)
    is used as the Sampath key when ``sampath`` is unset, for backward compat.

    The returned list drives round-robin distribution of pages across keys so that
    no single key is hammered while the others sit idle.
    """
    result: list[tuple[str, str]] = []
    sampath_key = _normalize_key(sampath) or _normalize_key(legacy)
    if sampath_key:
        result.append((sampath_key, "Sampath"))
    rishi_key = _normalize_key(rishi)
    if rishi_key:
        result.append((rishi_key, "Rishi"))
    syam_key = _normalize_key(syam)
    if syam_key:
        result.append((syam_key, "Syam"))
    return result
