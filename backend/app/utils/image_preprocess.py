"""Scan preprocessing for dots.ocr (from ``ocr_pipeline/preprocess.py``)."""
from __future__ import annotations

import io

from PIL import Image, ImageEnhance, ImageFilter


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _stretch_luma(rgb: Image.Image) -> Image.Image:
    import numpy as np

    ycbcr = rgb.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y_arr = np.asarray(y, dtype=np.float32)
    lo, hi = np.percentile(y_arr, (2.0, 98.0))
    span = max(float(hi - lo), 1.0)
    y_arr = np.clip((y_arr - lo) * (255.0 / span), 0, 255).astype(np.uint8)
    y2 = Image.fromarray(y_arr, mode="L")
    return Image.merge("YCbCr", (y2, cb, cr)).convert("RGB")


def _maybe_upscale(rgb: Image.Image, min_width: int = 1200, max_scale: float = 2.0) -> Image.Image:
    w, h = rgb.size
    if w >= min_width:
        return rgb
    scale = min(min_width / w, max_scale)
    if scale <= 1.0:
        return rgb
    nw, nh = int(w * scale), int(h * scale)
    return rgb.resize((nw, nh), Image.Resampling.LANCZOS)


def preprocess_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    *,
    enabled: bool,
) -> tuple[bytes, str]:
    """Return (bytes, mime_type) ready for dots.ocr (PNG when preprocessed)."""
    img = Image.open(io.BytesIO(image_bytes))
    out = _to_rgb(img)
    if enabled:
        out = _maybe_upscale(out)
        out = _stretch_luma(out)
        out = ImageEnhance.Contrast(out).enhance(1.12)
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    if mime_type in ("image/jpeg", "image/jpg"):
        return image_bytes, mime_type
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def preprocess_pil_image(img: Image.Image, *, enabled: bool) -> Image.Image:
    if not enabled:
        return _to_rgb(img)
    rgb = _to_rgb(img)
    rgb = _maybe_upscale(rgb)
    rgb = _stretch_luma(rgb)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.12)
    return rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
