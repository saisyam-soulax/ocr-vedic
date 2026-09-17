"""In-memory OCR job registry: active streams, cancel flags."""
from __future__ import annotations

import asyncio
import time
from typing import Any

_jobs: dict[str, tuple[asyncio.Queue, asyncio.Task | None, float]] = {}
_cancel_flags: dict[str, bool] = {}
JOB_TTL_SECONDS = 7200  # Drop finished registry entries after this age


def register_job(job_id: str, queue: asyncio.Queue, task: asyncio.Task) -> None:
    _jobs[job_id] = (queue, task, time.time())
    _cancel_flags.pop(job_id, None)


def drop_job(job_id: str) -> None:
    _jobs.pop(job_id, None)
    _cancel_flags.pop(job_id, None)


def get_job(job_id: str) -> tuple[asyncio.Queue, asyncio.Task | None, float] | None:
    return _jobs.get(job_id)


def is_job_active(job_id: str) -> bool:
    """True if job_id has a registered task that hasn't finished yet.

    Callers must check this before starting a new run (submit/resume) for a
    job_id — otherwise two overlapping tasks can run concurrently for the same
    job, and whichever finishes LAST silently overwrites the other's (possibly
    correct) completed results with its own status, corrupting the job's
    on-disk output. See INC-006 in docs/INCIDENT_LOG.md.
    """
    entry = _jobs.get(job_id)
    if entry is None:
        return False
    _, task, _ = entry
    return task is not None and not task.done()


def active_job_count() -> int:
    return len(_jobs)


def prune_stale_jobs(logger: Any) -> None:
    """Remove finished registry entries older than TTL.

    Never cancels an active OCR task — long multi-hour jobs must keep running
    even when newer submits trigger pruning.
    """
    now = time.time()
    stale: list[str] = []
    for k, (_, task, registered_at) in list(_jobs.items()):
        if now - registered_at <= JOB_TTL_SECONDS:
            continue
        if task is not None and not task.done():
            logger.debug(
                "Keeping active long-running job in registry: %s age=%.0fs",
                k,
                now - registered_at,
            )
            continue
        stale.append(k)
    for k in stale:
        _jobs.pop(k, None)
        _cancel_flags.pop(k, None)
        logger.info("Pruned stale in-memory job registry entry: %s", k)


def request_cancel(job_id: str) -> bool:
    if job_id not in _jobs:
        return False
    _cancel_flags[job_id] = True
    return True


def is_cancelled(job_id: str) -> bool:
    return _cancel_flags.get(job_id, False)
