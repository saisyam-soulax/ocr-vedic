"""Gemini API token pricing (USD per 1M tokens).

Rates follow Google AI Gemini API *Standard* paid tier as published at:
https://ai.google.dev/gemini-api/docs/pricing (retrieved 2026-05).

Formula (per API call):
  cost_usd = (input_tokens / 1_000_000) * input_rate
           + (output_tokens / 1_000_000) * output_rate

For models with a 200k prompt tier, the higher input/output rates apply when
prompt_token_count > 200_000 (per Google's "<= 200k" / "> 200k" columns).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_EFFECTIVE_LABEL = "2026-05 (Gemini API Standard paid tier)"
TIER_THRESHOLD_TOKENS = 200_000


@dataclass(frozen=True)
class _GeminiRates:
    """USD per 1M tokens."""

    input_le_200k: float
    output_le_200k: float
    input_gt_200k: float | None = None
    output_gt_200k: float | None = None

    def rates_for_prompt_size(self, prompt_tokens: int) -> tuple[float, float, str]:
        if (
            self.input_gt_200k is not None
            and self.output_gt_200k is not None
            and prompt_tokens > TIER_THRESHOLD_TOKENS
        ):
            return self.input_gt_200k, self.output_gt_200k, f">{TIER_THRESHOLD_TOKENS}"
        return self.input_le_200k, self.output_le_200k, f"<={TIER_THRESHOLD_TOKENS}"


# Longest-prefix wins when matching model ids (after normalization).
_MODEL_RATES: list[tuple[str, _GeminiRates]] = [
    # Gemini 3.1 / 3 Pro family (tiered at 200k)
    (
        "gemini-3.1-pro",
        _GeminiRates(2.00, 12.00, 4.00, 18.00),
    ),
    (
        "gemini-3-pro",
        _GeminiRates(2.00, 12.00, 4.00, 18.00),
    ),
    (
        "gemini-3.1-flash-lite",
        _GeminiRates(0.25, 1.50),
    ),
    (
        "gemini-3-flash",
        _GeminiRates(0.50, 3.00),
    ),
    # Gemini 2.5
    (
        "gemini-2.5-pro",
        _GeminiRates(1.25, 10.00, 2.50, 15.00),
    ),
    (
        "gemini-2.5-flash-lite",
        _GeminiRates(0.10, 0.40),
    ),
    (
        "gemini-2.5-flash",
        _GeminiRates(0.30, 2.50),
    ),
    # Gemini 2.0 (deprecated Jun 2026; kept for cost estimates on legacy jobs)
    (
        "gemini-2.0-flash-lite",
        _GeminiRates(0.075, 0.30),
    ),
    (
        "gemini-2.0-flash",
        _GeminiRates(0.10, 0.40),
    ),
    # Legacy 1.5 (historical jobs / aliases)
    (
        "gemini-1.5-pro",
        _GeminiRates(1.25, 5.00, 2.50, 10.00),
    ),
    (
        "gemini-1.5-flash",
        _GeminiRates(0.075, 0.30),
    ),
]

_DEFAULT_RATES = _GeminiRates(0.30, 2.50)  # gemini-2.5-flash fallback


def normalize_model_id(model: str) -> str:
    m = (model or "").strip().lower()
    m = re.sub(r"^models/", "", m)
    return m


def resolve_gemini_rates(model: str) -> tuple[_GeminiRates, str]:
    """Return rates and the pricing table key used."""
    normalized = normalize_model_id(model)
    for prefix, rates in _MODEL_RATES:
        if normalized == prefix or normalized.startswith(prefix + "-"):
            return rates, prefix
    return _DEFAULT_RATES, "gemini-2.5-flash (default)"


def estimate_gemini_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """Estimate USD cost for one generateContent call."""
    rates, pricing_key = resolve_gemini_rates(model)
    in_rate, out_rate, tier = rates.rates_for_prompt_size(max(0, input_tokens))
    input_tokens = max(0, int(input_tokens))
    output_tokens = max(0, int(output_tokens))
    input_cost = (input_tokens / 1_000_000) * in_rate
    output_cost = (output_tokens / 1_000_000) * out_rate
    total = input_cost + output_cost
    return {
        "pricing_model_key": pricing_key,
        "pricing_tier": tier,
        "input_cost_per_1m_usd": in_rate,
        "output_cost_per_1m_usd": out_rate,
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "estimated_cost_usd": round(total, 8),
        "pricing_source": PRICING_SOURCE_URL,
        "pricing_effective": PRICING_EFFECTIVE_LABEL,
    }
