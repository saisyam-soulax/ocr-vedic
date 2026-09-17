import asyncio
import logging
import time

import app.job_registry as reg
from app.job_registry import (
    JOB_TTL_SECONDS,
    active_job_count,
    is_cancelled,
    is_job_active,
    prune_stale_jobs,
    register_job,
    request_cancel,
)


def test_cancel_flag() -> None:
    async def _inner() -> None:
        reg._jobs.clear()
        reg._cancel_flags.clear()
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(asyncio.sleep(60))
        register_job("job-a", q, task)
        assert active_job_count() == 1
        assert request_cancel("job-a")
        assert is_cancelled("job-a")
        task.cancel()

    asyncio.run(_inner())


def test_prune_stale_does_not_cancel_active() -> None:
    async def _inner() -> None:
        reg._jobs.clear()
        reg._cancel_flags.clear()
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(asyncio.sleep(60))
        register_job("long-job", q, task)
        # Backdate registration beyond TTL while task is still running.
        queue, t, _ = reg._jobs["long-job"]
        reg._jobs["long-job"] = (queue, t, time.time() - JOB_TTL_SECONDS - 10)
        prune_stale_jobs(logging.getLogger("test"))
        assert "long-job" in reg._jobs
        assert not task.done()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_inner())


def test_is_job_active_true_while_task_running() -> None:
    async def _inner() -> None:
        reg._jobs.clear()
        reg._cancel_flags.clear()
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(asyncio.sleep(60))
        register_job("running-job", q, task)
        assert is_job_active("running-job")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_inner())


def test_is_job_active_false_once_task_finishes() -> None:
    async def _inner() -> None:
        reg._jobs.clear()
        reg._cancel_flags.clear()
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(asyncio.sleep(0))
        await task
        register_job("finished-job", q, task)
        assert not is_job_active("finished-job")

    asyncio.run(_inner())


def test_is_job_active_false_when_never_registered() -> None:
    reg._jobs.clear()
    reg._cancel_flags.clear()
    assert not is_job_active("never-seen-job")


def test_prune_stale_drops_finished_entries() -> None:
    async def _inner() -> None:
        reg._jobs.clear()
        reg._cancel_flags.clear()
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(asyncio.sleep(0))
        await task
        register_job("done-job", q, task)
        queue, t, _ = reg._jobs["done-job"]
        reg._jobs["done-job"] = (queue, t, time.time() - JOB_TTL_SECONDS - 10)
        prune_stale_jobs(logging.getLogger("test"))
        assert "done-job" not in reg._jobs

    asyncio.run(_inner())
