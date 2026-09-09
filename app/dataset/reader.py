"""
Read-only access to the Stage 2 annotated dataset (ULCA-schema JSON + images).

Everything here reads from ANNOTATED_DIR; nothing in this module ever writes
back into the dataset repo.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import ANNOTATED_DIR, ANNOTATED_SPLITS, SYNTHETIC_SAMPLE_CAP


class SampleNotFoundError(Exception):
    pass


def list_scripts() -> list[str]:
    """Scripts that have at least one annotated sample, across any split."""
    scripts: set[str] = set()
    for split in ANNOTATED_SPLITS:
        split_dir = ANNOTATED_DIR / split
        if not split_dir.is_dir():
            continue
        for script_dir in split_dir.iterdir():
            if script_dir.is_dir():
                scripts.add(script_dir.name)
    return sorted(scripts)


def _script_dirs(script: str):
    for split in ANNOTATED_SPLITS:
        script_dir = ANNOTATED_DIR / split / script
        if script_dir.is_dir():
            yield split, script_dir


def _is_synthetic(record: dict) -> bool:
    return any("synthetic" in s.lower() for s in record.get("collectionSource", []))


@lru_cache(maxsize=64)
def list_samples(script: str) -> list[dict]:
    """One entry per image for a script: {split, filename, fieldCount, sourceLanguage}.
    Synthetic (template-generated) samples are capped at SYNTHETIC_SAMPLE_CAP;
    real-world samples are never capped -- see app/config.py."""
    samples = []
    synthetic_count = 0
    for split, script_dir in _script_dirs(script):
        for json_path in sorted(script_dir.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not records:
                continue
            image_filename = records[0]["imageFilename"]
            if not (script_dir / image_filename).exists():
                continue

            is_synthetic = _is_synthetic(records[0])
            if is_synthetic:
                synthetic_count += 1
                if synthetic_count > SYNTHETIC_SAMPLE_CAP:
                    continue

            samples.append({
                "split": split,
                "filename": image_filename,
                "fieldCount": len(records),
                "sourceLanguage": records[0]["languages"].get("sourceLanguageName"),
                "imageTextType": records[0].get("imageTextType"),
                "isSynthetic": is_synthetic,
            })
    return samples


def _safe_component(value: str, label: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise SampleNotFoundError(f"Invalid {label}: {value!r}")
    return value


def get_records(script: str, split: str, image_filename: str) -> list[dict]:
    """All field-level ULCA records for one image (script + boundingBox + groundTruth)."""
    script = _safe_component(script, "script")
    split = _safe_component(split, "split")
    image_filename = _safe_component(image_filename, "image_filename")
    script_dir = ANNOTATED_DIR / split / script
    json_path = script_dir / (Path(image_filename).stem + ".json")
    if not json_path.exists():
        raise SampleNotFoundError(f"No records for {script}/{split}/{image_filename}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(script: str, split: str, image_filename: str) -> Path:
    script = _safe_component(script, "script")
    split = _safe_component(split, "split")
    image_filename = _safe_component(image_filename, "image_filename")
    image_path = ANNOTATED_DIR / split / script / image_filename
    if not image_path.exists():
        raise SampleNotFoundError(f"No image {script}/{split}/{image_filename}")
    return image_path
