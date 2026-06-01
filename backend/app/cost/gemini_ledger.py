"""Persistent Gemini OCR cost logs (per job + global administrator ledger)."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.cost.gemini_pricing import estimate_gemini_cost_usd

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.Lock()


@dataclass
class GeminiCostSession:
    """Accumulates token usage for one OCR job (Gemini provider only)."""

    job_id: str
    batch_dir: Path
    model: str
    provider: str = "gemini"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    calls: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def record_call(
        self,
        *,
        page_index: int,
        page_in_source: int | None,
        source_file: str,
        step: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        cost = estimate_gemini_cost_usd(self.model, input_tokens, output_tokens)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "job_id": self.job_id,
            "page_index": page_index,
            "page_in_source": page_in_source,
            "source_file": source_file,
            "step": step,
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
            **cost,
        }
        self.calls.append(entry)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def build_job_log(self, *, elapsed_seconds: float | None = None) -> dict[str, Any]:
        summary_cost = estimate_gemini_cost_usd(
            self.model, self.total_input_tokens, self.total_output_tokens
        )
        ended = datetime.now(UTC)
        return {
            "session_info": {
                "job_id": self.job_id,
                "provider": self.provider,
                "model": self.model,
                "start_time": self.started_at.isoformat(),
                "end_time": ended.isoformat(),
                "duration_seconds": (ended - self.started_at).total_seconds(),
                "ocr_elapsed_seconds": elapsed_seconds,
            },
            "summary": {
                "total_api_calls": len(self.calls),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "estimated_total_cost_usd": summary_cost["estimated_cost_usd"],
                "input_cost_usd": summary_cost["input_cost_usd"],
                "output_cost_usd": summary_cost["output_cost_usd"],
                "pricing_model_key": summary_cost["pricing_model_key"],
                "pricing_tier": summary_cost["pricing_tier"],
                "input_cost_per_1m_usd": summary_cost["input_cost_per_1m_usd"],
                "output_cost_per_1m_usd": summary_cost["output_cost_per_1m_usd"],
                "pricing_source": summary_cost["pricing_source"],
                "pricing_effective": summary_cost["pricing_effective"],
            },
            "api_calls": self.calls,
        }

    def save_job_log(self, *, elapsed_seconds: float | None = None) -> Path:
        log = self.build_job_log(elapsed_seconds=elapsed_seconds)
        path = self.batch_dir / "gemini_cost_log.json"
        path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Gemini cost log saved: job=%s calls=%d total_usd=%.6f path=%s",
            self.job_id,
            len(self.calls),
            log["summary"]["estimated_total_cost_usd"],
            path,
        )
        return path

    def append_to_global_ledger(self, ledger_path: Path, *, elapsed_seconds: float | None) -> None:
        """Append job summary + per-call lines to administrator JSONL ledger."""
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        job_log = self.build_job_log(elapsed_seconds=elapsed_seconds)
        summary_record = {
            "record_type": "job_summary",
            "timestamp": datetime.now(UTC).isoformat(),
            **job_log["session_info"],
            **job_log["summary"],
        }
        with _WRITE_LOCK:
            with ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
                for call in self.calls:
                    fh.write(
                        json.dumps(
                            {"record_type": "api_call", **call},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


def start_gemini_cost_session(
    *,
    job_id: str,
    batch_dir: Path,
    model: str,
) -> GeminiCostSession:
    return GeminiCostSession(job_id=job_id, batch_dir=batch_dir, model=model)


def finalize_gemini_job_costs(
    session: GeminiCostSession | None,
    *,
    ledger_path: Path,
    elapsed_seconds: float | None,
) -> Path | None:
    if session is None or not session.calls:
        return None
    job_path = session.save_job_log(elapsed_seconds=elapsed_seconds)
    session.append_to_global_ledger(ledger_path, elapsed_seconds=elapsed_seconds)
    return job_path


def read_global_ledger(
    ledger_path: Path,
    *,
    limit: int = 200,
    job_id: str | None = None,
) -> list[dict[str, Any]]:
    if not ledger_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if job_id and rec.get("job_id") != job_id:
            continue
        records.append(rec)
    if job_id:
        return records[-limit:]
    # Default: job summaries only for list view
    summaries = [r for r in records if r.get("record_type") == "job_summary"]
    return summaries[-limit:]
