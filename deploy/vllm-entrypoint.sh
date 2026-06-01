#!/usr/bin/env bash
# Auto-tune --gpu-memory-utilization from free VRAM (shared GPU, no other jobs killed).
# Set VLLM_GPU_MEMORY_UTILIZATION=auto in .env, or a fixed value like 0.03.
# GB10 unified memory: nvidia-smi often reports [N/A]; fall back to torch.cuda.mem_get_info.
set -euo pipefail

compute_util() {
  python3 <<'PY'
import subprocess
import sys

ratio = None
source = "nvidia-smi"
free_b = total_b = None

try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()[0]
    parts = [x.strip() for x in out.split(",")[:2]]
    if not any("N/A" in p.upper() or p == "" for p in parts):
        free_b = float(parts[0]) * 1024 * 1024
        total_b = float(parts[1]) * 1024 * 1024
        ratio = free_b / total_b if total_b > 0 else None
except (subprocess.CalledProcessError, ValueError, IndexError):
    pass

if ratio is None:
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            ratio = free_b / total_b if total_b > 0 else None
            source = "torch.cuda.mem_get_info"
    except Exception:
        pass

if ratio is None or free_b is None or total_b is None:
    fallback = float(__import__("os").environ.get("VLLM_GPU_MEMORY_UTILIZATION_FALLBACK", "0.025"))
    print(f"[vllm-entrypoint] could not read GPU memory; using fallback {fallback:.3f}", file=sys.stderr)
    print(f"{fallback:.3f}")
    sys.exit(0)

# vLLM checks (util * total) <= free when EngineCore starts. APIServer uses ~1-2 GiB
# between this probe and engine init, so budget from (free - overhead) with headroom.
overhead_b = int(
    __import__("os").environ.get("VLLM_AUTO_MEMORY_OVERHEAD_BYTES", str(2 * 1024**3))
)
budget_b = max(0, free_b - overhead_b)
# 95% headroom: engine must see free >= util * total
util = min(0.45, max(0.01, (budget_b / total_b) * 0.95))
print(
    f"[vllm-entrypoint] auto source={source} free={free_b/1024**3:.2f}GiB "
    f"overhead={overhead_b/1024**3:.2f}GiB budget={budget_b/1024**3:.2f}GiB util={util:.3f}",
    file=sys.stderr,
)
print(f"{util:.3f}")
PY
}

UTIL_SETTING="${VLLM_GPU_MEMORY_UTILIZATION:-auto}"
if [ "$UTIL_SETTING" = "auto" ]; then
  UTIL="$(compute_util)"
  echo "[vllm-entrypoint] auto gpu-memory-utilization=${UTIL}"
else
  UTIL="$UTIL_SETTING"
fi

# Replace --gpu-memory-utilization <value> in compose command args
NEW_ARGS=()
skip_next=0
for arg in "$@"; do
  if [ "$skip_next" = "1" ]; then
    NEW_ARGS+=("$UTIL")
    skip_next=0
    continue
  fi
  if [ "$arg" = "--gpu-memory-utilization" ]; then
    NEW_ARGS+=("$arg")
    skip_next=1
    continue
  fi
  NEW_ARGS+=("$arg")
done

exec vllm "${NEW_ARGS[@]}"
