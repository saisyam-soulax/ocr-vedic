"""Aggregate Gemini cost logs from upload batches."""

import json
from pathlib import Path

from app.cost.gemini_ledger import aggregate_gemini_costs, read_job_cost_summary


def test_read_job_cost_summary_missing(tmp_path: Path) -> None:
    assert read_job_cost_summary(tmp_path) is None


def test_aggregate_gemini_costs_empty(tmp_path: Path) -> None:
    out = aggregate_gemini_costs(tmp_path)
    assert out["job_count"] == 0
    assert out["estimated_total_cost_usd"] == 0.0
    assert out["jobs"] == []


def test_aggregate_sums_job_logs(tmp_path: Path) -> None:
    batch = tmp_path / "job-a"
    batch.mkdir()
    log = {
        "session_info": {"job_id": "job-a", "model": "gemini-2.5-flash"},
        "summary": {
            "total_api_calls": 2,
            "total_input_tokens": 1000,
            "total_output_tokens": 200,
            "input_cost_usd": 0.001,
            "output_cost_usd": 0.002,
            "estimated_total_cost_usd": 0.003,
            "pricing_source": "https://example.com",
            "pricing_effective": "test",
        },
        "api_calls": [],
    }
    (batch / "gemini_cost_log.json").write_text(json.dumps(log), encoding="utf-8")
    (batch / "metadata.json").write_text(
        json.dumps({"created_at": "2026-01-01T00:00:00+00:00", "files": ["a.pdf"]}),
        encoding="utf-8",
    )

    out = aggregate_gemini_costs(tmp_path)
    assert out["job_count"] == 1
    assert out["total_pages"] == 2
    assert out["estimated_total_cost_usd"] == 0.003
    assert out["jobs"][0]["job_id"] == "job-a"
