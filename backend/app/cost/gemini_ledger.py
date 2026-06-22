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


def cost_summary_from_job_log(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a gemini_cost_log.json payload to a compact summary dict."""
    session = data.get("session_info") or {}
    summary = data.get("summary") or {}
    return {
        "model": session.get("model"),
        "total_api_calls": int(summary.get("total_api_calls", 0) or 0),
        "total_input_tokens": int(summary.get("total_input_tokens", 0) or 0),
        "total_output_tokens": int(summary.get("total_output_tokens", 0) or 0),
        "input_cost_usd": float(summary.get("input_cost_usd", 0) or 0),
        "output_cost_usd": float(summary.get("output_cost_usd", 0) or 0),
        "estimated_total_cost_usd": float(summary.get("estimated_total_cost_usd", 0) or 0),
        "pricing_model_key": summary.get("pricing_model_key"),
        "pricing_source": summary.get("pricing_source"),
        "pricing_effective": summary.get("pricing_effective"),
    }


def read_job_cost_summary(batch_dir: Path) -> dict[str, Any] | None:
    """Read per-job Gemini cost summary from disk, if present."""
    cost_path = batch_dir / "gemini_cost_log.json"
    if not cost_path.is_file():
        return None
    try:
        data = json.loads(cost_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return cost_summary_from_job_log(data)


def aggregate_gemini_costs(upload_root: Path) -> dict[str, Any]:
    """Sum all gemini_cost_log.json files under the uploads directory."""
    totals = {
        "job_count": 0,
        "total_pages": 0,
        "total_api_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "estimated_total_cost_usd": 0.0,
    }
    by_model: dict[str, dict[str, Any]] = {}
    jobs: list[dict[str, Any]] = []

    if not upload_root.is_dir():
        return {**totals, "by_model": [], "jobs": [], "pricing_source": None, "pricing_effective": None}

    cost_paths = sorted(
        upload_root.glob("*/gemini_cost_log.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    pricing_source: str | None = None
    pricing_effective: str | None = None

    for cost_path in cost_paths:
        try:
            data = json.loads(cost_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = cost_summary_from_job_log(data)
        if summary["total_api_calls"] <= 0:
            continue

        job_id = (data.get("session_info") or {}).get("job_id") or cost_path.parent.name
        model = summary.get("model") or "unknown"
        pages = summary["total_api_calls"]

        totals["job_count"] += 1
        totals["total_pages"] += pages
        totals["total_api_calls"] += summary["total_api_calls"]
        totals["total_input_tokens"] += summary["total_input_tokens"]
        totals["total_output_tokens"] += summary["total_output_tokens"]
        totals["input_cost_usd"] += summary["input_cost_usd"]
        totals["output_cost_usd"] += summary["output_cost_usd"]
        totals["estimated_total_cost_usd"] += summary["estimated_total_cost_usd"]

        if summary.get("pricing_source"):
            pricing_source = summary["pricing_source"]
        if summary.get("pricing_effective"):
            pricing_effective = summary["pricing_effective"]

        bucket = by_model.setdefault(
            model,
            {
                "model": model,
                "job_count": 0,
                "total_pages": 0,
                "total_api_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "estimated_total_cost_usd": 0.0,
            },
        )
        bucket["job_count"] += 1
        bucket["total_pages"] += pages
        bucket["total_api_calls"] += summary["total_api_calls"]
        bucket["total_input_tokens"] += summary["total_input_tokens"]
        bucket["total_output_tokens"] += summary["total_output_tokens"]
        bucket["input_cost_usd"] += summary["input_cost_usd"]
        bucket["output_cost_usd"] += summary["output_cost_usd"]
        bucket["estimated_total_cost_usd"] += summary["estimated_total_cost_usd"]

        submitted_at = None
        files: list[str] = []
        meta_path = cost_path.parent / "metadata.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                submitted_at = meta.get("created_at")
                files = meta.get("files") or []
            except (json.JSONDecodeError, OSError):
                pass

        jobs.append(
            {
                "job_id": job_id,
                "model": model,
                "submitted_at": submitted_at,
                "files": files,
                **summary,
            }
        )

    by_model_list = sorted(
        by_model.values(),
        key=lambda x: x["estimated_total_cost_usd"],
        reverse=True,
    )
    return {
        **totals,
        "input_cost_usd": round(totals["input_cost_usd"], 6),
        "output_cost_usd": round(totals["output_cost_usd"], 6),
        "estimated_total_cost_usd": round(totals["estimated_total_cost_usd"], 6),
        "pricing_source": pricing_source,
        "pricing_effective": pricing_effective,
        "by_model": by_model_list,
        "jobs": jobs,
    }
