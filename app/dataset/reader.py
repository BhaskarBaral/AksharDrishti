"""
Read-only access to the demo's word-level test set:
TestData/<script>/word_level/ (real handwritten word crops + manifest.json,
built by TestData/_tools/build_testdata.py, all held out from the fine-tuned
recognizers' training data).

Each crop is exposed as a one-record ULCA-shaped sample whose boundingBox is
the whole image, so the OCR tab's field-sample flow (run-on-sample,
benchmark) works on it unchanged. The `split` path component is always
"word_level". Nothing here ever writes.
"""
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image

from app.config import TESTDATA_DIR, TESTDATA_SCRIPTS

SPLIT = "word_level"


class SampleNotFoundError(Exception):
    pass


def list_scripts() -> list[str]:
    """Scripts that have a word-level test folder, in the demo's display order."""
    return [s for s in TESTDATA_SCRIPTS if (TESTDATA_DIR / s / SPLIT / "manifest.json").exists()]


@lru_cache(maxsize=64)
def _manifest(script: str) -> dict:
    path = TESTDATA_DIR / script / SPLIT / "manifest.json"
    if not path.exists():
        return {}
    return {e["file"]: e for e in json.loads(path.read_text(encoding="utf-8"))}


def list_samples(script: str) -> list[dict]:
    """One entry per word crop: {split, filename, fieldCount, imageTextType, isSynthetic, humanReviewed}."""
    return [{
        "split": SPLIT,
        "filename": name,
        "fieldCount": 1,
        "imageTextType": "handwritten-text",
        "isSynthetic": e.get("source") != "real",
        "humanReviewed": e.get("humanReviewed", False),
    } for name, e in _manifest(script).items()]


def _safe_component(value: str, label: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise SampleNotFoundError(f"Invalid {label}: {value!r}")
    return value


def resolve_image_path(script: str, split: str, image_filename: str) -> Path:
    script = _safe_component(script, "script")
    image_filename = _safe_component(image_filename, "image_filename")
    if split != SPLIT or image_filename not in _manifest(script):
        raise SampleNotFoundError(f"No image {script}/{split}/{image_filename}")
    return TESTDATA_DIR / script / SPLIT / image_filename


def get_records(script: str, split: str, image_filename: str) -> list[dict]:
    """The crop's single record: whole-image boundingBox + groundTruth."""
    path = resolve_image_path(script, split, image_filename)
    entry = _manifest(script)[image_filename]
    w, h = Image.open(path).size
    return [{
        "imageFilename": image_filename,
        "groundTruth": entry["groundTruth"],
        "imageTextType": "handwritten-text",
        "collectionSource": entry.get("collectionSource") or [],
        "boundingBox": {"vertices": [{"x": 0, "y": 0}, {"x": w, "y": 0}, {"x": w, "y": h}, {"x": 0, "y": h}]},
    }]
