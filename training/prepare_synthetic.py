"""
Mix synthetic samples into train.txt for specific scripts -- Stage 4 phase 2,
a targeted pilot on a few scripts before deciding whether to roll it out
broadly (see CHARTER.md; prepare_datasets.py's phase-1 run was real data
only, by design, so this gain can be measured against that real-only
baseline). Only appends to train.txt; val.txt stays real-only (built by
prepare_datasets.py) so the eval number is still comparable to phase 1.

Synthetic images are full multi-field *documents*
(synthetic/<script>/<doctype>/<tier>/*.json+image), unlike raw-data/'s
pre-cropped single-word images -- each field needs a bbox crop first
(same convention as app/imaging.py's bbox_crop: min/max over vertices,
padded).

Held-out honesty: the annotated/ folder's *synthetic* fields are drawn from
this exact synthetic/ pool (verified by filename: e.g.
annotated/test/tamil/taml_hw_clean_0293.png is the same file as
synthetic/tamil/hw/clean/taml_hw_clean_0293.png). demo/rebenchmark_all.py
scores every engine's synthetic-CER against those fields, so every source
document that has ANY field curated into annotated/ is excluded here
entirely (not just the specific field-index used there) -- conservative,
but guarantees the synthetic-CER column stays a genuine holdout.

Run (after prepare_datasets.py + generate_configs.py have already run once):
  .venv\\Scripts\\python.exe training/prepare_synthetic.py devanagari tamil malayalam
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image

from prepare_datasets import ANNOTATED_DIR, DATASET_REPO, DATASETS_OUT, OFFICIAL_DICT, sanitize_label

SYNTHETIC_DIR = DATASET_REPO / "synthetic"
SEED = 42
CROP_PAD_PX = 8
# Roughly matches phase-1's real train_n (887/script) rather than letting
# synthetic's much larger, narrower-vocabulary pool dominate the mix.
SYNTHETIC_CAP = 900


def bbox_crop(im: Image.Image, vertices: list[dict], pad: int = CROP_PAD_PX) -> Image.Image:
    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    left = max(0, min(xs) - pad)
    top = max(0, min(ys) - pad)
    right = min(im.width, max(xs) + pad)
    bottom = min(im.height, max(ys) + pad)
    return im.crop((left, top, right, bottom))


def load_held_out_synthetic_filenames(script: str) -> set[str]:
    held_out = set()
    for split in ("train", "val", "test"):
        script_dir = ANNOTATED_DIR / split / script
        if not script_dir.is_dir():
            continue
        for json_path in script_dir.glob("*.json"):
            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                if any("synthetic" in s.lower() for s in rec.get("collectionSource", [])):
                    held_out.add(rec["imageFilename"])
    return held_out


def collect_field_records(script: str, held_out: set[str]) -> list[tuple[Path, int, list[dict], str]]:
    """(doc_image_path, field_index, bbox_vertices, label) across every
    doctype/tier, skipping any document with a held-out filename."""
    script_dir = SYNTHETIC_DIR / script
    fields = []
    for json_path in sorted(script_dir.glob("*/*/*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if not records or records[0]["imageFilename"] in held_out:
            continue
        image_path = json_path.with_suffix(f".{records[0]['format']}")
        if not image_path.exists():
            continue
        for i, rec in enumerate(records):
            gt = (rec.get("groundTruth") or "").strip()
            if gt:
                fields.append((image_path, i, rec["boundingBox"]["vertices"], gt))
    return fields


def build_synthetic_train_additions(script: str) -> dict:
    held_out = load_held_out_synthetic_filenames(script)
    fields = collect_field_records(script, held_out)

    rng = random.Random(SEED)
    rng.shuffle(fields)
    fields = fields[:SYNTHETIC_CAP]

    out_dir = DATASETS_OUT / script
    crops_dir = out_dir / "synthetic_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    open_images: dict[Path, Image.Image] = {}
    lines = []
    for doc_path, field_idx, vertices, label in fields:
        if doc_path not in open_images:
            open_images[doc_path] = Image.open(doc_path).convert("RGB")
        crop = bbox_crop(open_images[doc_path], vertices)
        crop_path = crops_dir / f"{doc_path.stem}_{field_idx}.png"
        crop.save(crop_path)
        lines.append(f"{crop_path.resolve()}\t{sanitize_label(label)}\n")

    train_path = out_dir / "train.txt"
    with open(train_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

    dict_added = 0
    if script not in OFFICIAL_DICT:
        # Custom dicts (built from prepare_datasets.py's real-only label set)
        # won't cover synthetic vocabulary -- form-template digits/Latin/
        # punctuation, and sometimes genuine script characters that never
        # happened to appear in the ~1,000-sample real pool. Extend it so
        # training doesn't silently corrupt or crash on out-of-dict labels.
        dict_path = out_dir / "dict.txt"
        existing = dict_path.read_text(encoding="utf-8").splitlines()
        existing_set = set(existing)
        new_chars = set()
        for _, _, _, label in fields:
            new_chars.update(c for c in label if c != " ")
        added = sorted(new_chars - existing_set, key=ord)
        if added:
            with open(dict_path, "a", encoding="utf-8") as f:
                for c in added:
                    f.write(c + "\n")
        dict_added = len(added)

    return {
        "script": script,
        "synthetic_added": len(lines),
        "docs_excluded": len(held_out),
        "dict_chars_added": dict_added,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scripts", nargs="+")
    args = parser.parse_args()

    for script in args.scripts:
        result = build_synthetic_train_additions(script)
        print(
            f"{result['script']}: +{result['synthetic_added']} synthetic samples "
            f"appended to train.txt ({result['docs_excluded']} held-out docs excluded, "
            f"+{result['dict_chars_added']} dict chars added)"
        )


if __name__ == "__main__":
    main()
