"""Gemini cost estimation tests (pricing table + formula)."""

from app.cost.gemini_pricing import (
    TIER_THRESHOLD_TOKENS,
    estimate_gemini_cost_usd,
    resolve_gemini_rates,
)


def test_gemini_25_flash_pricing() -> None:
    rates, key = resolve_gemini_rates("gemini-2.5-flash")
    assert key == "gemini-2.5-flash"
    assert rates.input_le_200k == 0.30
    assert rates.output_le_200k == 2.50


def test_gemini_25_pro_tiered_pricing() -> None:
    small = estimate_gemini_cost_usd("gemini-2.5-pro", 20_000, 10_000)
    assert small["pricing_tier"] == f"<={TIER_THRESHOLD_TOKENS}"
    assert small["input_cost_per_1m_usd"] == 1.25
    assert small["output_cost_per_1m_usd"] == 10.00
    # (0.02 * 1.25) + (0.01 * 10) = 0.025 + 0.10 = 0.125
    assert abs(small["estimated_cost_usd"] - 0.125) < 1e-6

    large = estimate_gemini_cost_usd("gemini-2.5-pro", 250_000, 10_000)
    assert large["pricing_tier"] == f">{TIER_THRESHOLD_TOKENS}"
    assert large["input_cost_per_1m_usd"] == 2.50
    assert large["output_cost_per_1m_usd"] == 15.00


def test_gemini_31_pro_preview_alias() -> None:
    cost = estimate_gemini_cost_usd("gemini-3.1-pro-preview-customtools", 10_000, 5_000)
    assert cost["input_cost_per_1m_usd"] == 2.00
    assert cost["output_cost_per_1m_usd"] == 12.00


def test_cost_formula_matches_reference_style() -> None:
    """Same structure as pdfToGeminiRangeSpecific.CostTracker.calculate_cost."""
    input_tokens = 50_000
    output_tokens = 10_000
    cost = estimate_gemini_cost_usd("gemini-2.5-flash", input_tokens, output_tokens)
    manual = (input_tokens / 1_000_000) * 0.30 + (output_tokens / 1_000_000) * 2.50
    assert abs(cost["estimated_cost_usd"] - round(manual, 8)) < 1e-9
