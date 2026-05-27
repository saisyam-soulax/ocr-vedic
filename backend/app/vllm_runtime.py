"""On-demand vLLM lifecycle management.

State machine:  stopped → starting → ready → stopping → stopped

When VLLM_ON_DEMAND=false the load/unload calls just poll health without
touching Docker; this is still useful so the UI can detect when an
externally-started vLLM becomes ready.

The model is started/stopped *manually* via the UI (/api/vllm/load and
/api/vllm/unload).  There is no auto-idle shutdown — the user controls when
the GPU is released.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import httpx

from app.schemas import VllmState

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)

# Module-level state — single event loop, single worker process.
_state: VllmState = VllmState.stopped
_error_msg: str | None = None
_load_task: asyncio.Task | None = None
_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vllm_health_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return f"{root.rstrip('/')}/health"


async def check_health(base_url: str, timeout: float = 5.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(_vllm_health_url(base_url))
            return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_state() -> VllmState:
    return _state


def get_error() -> str | None:
    return _error_msg


async def load(settings: Settings) -> None:
    """Start loading vLLM (non-blocking — polls in background task)."""
    global _state, _error_msg, _load_task

    async with _lock:
        if _state in (VllmState.ready, VllmState.starting):
            return
        _state = VllmState.starting
        _error_msg = None
        if _load_task and not _load_task.done():
            _load_task.cancel()
        _load_task = asyncio.create_task(_do_load(settings))


async def unload(settings: Settings) -> None:
    global _state

    async with _lock:
        if _state in (VllmState.stopped, VllmState.stopping):
            return  # already stopped or a stop is already in flight
        _state = VllmState.stopping

    if settings.vllm_on_demand:
        await _docker_stop(settings.vllm_container_name)

    async with _lock:
        _state = VllmState.stopped


# ---------------------------------------------------------------------------
# Internal coroutines
# ---------------------------------------------------------------------------

async def _do_load(settings: Settings) -> None:
    global _state, _error_msg

    if settings.vllm_on_demand:
        await _docker_start(settings.vllm_container_name)

    # Poll health until ready or 10-minute deadline.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if await check_health(settings.vllm_base_url):
            async with _lock:
                _state = VllmState.ready
            logger.info("vLLM is ready")
            if settings.vllm_warmup_on_load:
                asyncio.create_task(_warmup(settings))
            return
        await asyncio.sleep(5)

    async with _lock:
        _state = VllmState.error
        _error_msg = "vLLM did not become healthy within 10 minutes"
    logger.error("vLLM health-check timed out")


async def _docker_start(container_name: str) -> None:
    """Start a container via the Docker socket API."""
    sock = "/var/run/docker.sock"
    if not __import__("os").path.exists(sock):
        logger.warning("Docker socket not found at %s — skipping container start", sock)
        return
    try:
        transport = httpx.AsyncHTTPTransport(uds=sock)
        async with httpx.AsyncClient(transport=transport, timeout=30) as client:
            r = await client.post(f"http://docker/containers/{container_name}/start")
            if r.status_code in (204, 304):
                logger.info("Container %s started (status=%d)", container_name, r.status_code)
            else:
                logger.warning(
                    "docker start %s returned %d: %s",
                    container_name, r.status_code, r.text[:300],
                )
    except Exception as exc:
        logger.warning("docker start %s failed: %s", container_name, exc)


async def _docker_stop(container_name: str) -> None:
    """Stop a container via the Docker socket API.

    Uses /stop?t=0 which:
      • Sends SIGKILL *immediately* (no graceful-shutdown wait).
      • Sets Docker's "explicitly stopped" flag so restart: unless-stopped does
        NOT restart the container (unlike /kill which skips that flag).
      • Waits for the container to exit before returning 204.  With SIGKILL the
        process exits in <1 s; 60 s timeout is ample.
    """
    sock = "/var/run/docker.sock"
    if not __import__("os").path.exists(sock):
        logger.warning("Docker socket not found — skipping container stop")
        return
    try:
        transport = httpx.AsyncHTTPTransport(uds=sock)
        async with httpx.AsyncClient(transport=transport, timeout=60) as client:
            r = await client.post(
                f"http://docker/containers/{container_name}/stop",
                params={"t": "0"},   # t=0 → immediate SIGKILL, no grace period
            )
            if r.status_code in (204, 304):
                logger.info("Container %s stopped (status=%d)", container_name, r.status_code)
            elif r.status_code == 404:
                logger.info("Container %s not found — already removed", container_name)
            elif r.status_code == 409:
                logger.info("Container %s already stopped (409)", container_name)
            else:
                logger.warning(
                    "docker stop %s returned %d: %s",
                    container_name, r.status_code, r.text[:300],
                )
    except Exception as exc:
        logger.warning("docker stop %s failed: %s", container_name, exc)


async def _warmup(settings: Settings) -> None:
    """Fire a cheap text-only request to trigger Triton JIT before first real job."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            payload = {
                "model": settings.vllm_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
                "temperature": 0,
            }
            r = await client.post(
                f"{settings.vllm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {settings.vllm_api_key or 'EMPTY'}"},
            )
            logger.info("vLLM warmup complete: status=%d", r.status_code)
    except Exception as exc:
        logger.info("vLLM warmup skipped: %s", exc)
