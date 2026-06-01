"""Load vLLM / dots.ocr system prompt from ``prompt.txt`` (``ocr_pipeline/prompt_loader.py``)."""
from __future__ import annotations

import re
from pathlib import Path


def load_prompt_from_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    m = re.search(r"PROMPT\s*=\s*'''(.*)'''\s*\Z", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    # Plain text file
    return raw.strip()
