"""Bounding-box cropping and text-accuracy scoring shared by the OCR router."""
from PIL import Image

from app.config import CROP_PAD_PX


def bbox_crop(im: Image.Image, vertices: list[dict], pad: int = CROP_PAD_PX) -> Image.Image:
    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    left, top = max(0, min(xs) - pad), max(0, min(ys) - pad)
    right, bottom = min(im.width, max(xs) + pad), min(im.height, max(ys) + pad)
    return im.crop((left, top, right, bottom))


def safe_cer(ground_truth: str, hypothesis: str) -> float | None:
    if not ground_truth.strip():
        return None
    if hypothesis.startswith("__ERROR__") or not hypothesis.strip():
        return 1.0
    import jiwer
    try:
        return round(jiwer.cer(ground_truth, hypothesis), 4)
    except Exception:
        return 1.0
