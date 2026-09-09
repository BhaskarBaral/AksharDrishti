"""
Build PaddleX/PaddleOCR-format recognition training data for every script that
has real (non-synthetic) data, from the sibling dataset repo's untapped
`raw-data/` pool -- ~1,000 real handwritten word-crops per script, read-only
(same convention as everything else that touches Akshar_dristhii_code).

Stage 4, phase 1: REAL DATA ONLY. Synthetic augmentation is a deliberate later
phase (see demo/CHARTER.md) -- mixing it in now would make it impossible to
tell whether an accuracy gain came from real signal or from the synthetic
templates' narrower vocabulary.

Held-out honesty: the annotated/{train,val,test} real fields per script are
the exact set `demo/rebenchmark_all.py` already scored the 4 off-the-shelf
engines against. Every one of those images is excluded from the training
pool here (by filename) so the fine-tuned model's eventual eval on that same
set is a genuine before/after comparison, not train-on-test.

Per-script recognition head/dict strategy:
  - devanagari, tamil, telugu, urdu: an official PP-OCRv5 mobile rec
    checkpoint exists for this exact script (urdu uses the Arabic checkpoint
    -- Perso-Arabic script family match). Reuse that checkpoint's own
    official dictionary (bundled with the PaddleOCR training repo) so the
    classifier head shape matches exactly and the FULL pretrained weights
    load, not just the backbone. Verified zero out-of-dictionary characters
    against our real label set before relying on this (see
    training/_dict_check.txt).
  - bengali, gujarati, kannada, malayalam, odia, punjabi: no official PP-OCRv5
    checkpoint exists for these scripts at all. Dictionary is built from this
    script's own real label characters; initialized from the Devanagari
    checkpoint as a generic Indic-abugida backbone (closest available shared
    visual heritage), with the classifier head re-initialized -- standard
    cross-script transfer-learning practice when no matching checkpoint
    exists.

Run:
  .venv\\Scripts\\python.exe training/prepare_datasets.py
"""
import json
import random
import shutil
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = Path(__file__).resolve().parent
DATASET_REPO = DEMO_ROOT.parent.parent / "Akshar_dristhii_code"
RAW_DATA_DIR = DATASET_REPO / "raw-data"
ANNOTATED_DIR = DATASET_REPO / "annotated"

PADDLEOCR_REPO = (
    DEMO_ROOT / ".venv" / "Lib" / "site-packages" / "paddlex"
    / "repo_manager" / "repos" / "PaddleOCR"
)
OFFICIAL_DICT_DIR = PADDLEOCR_REPO / "ppocr" / "utils" / "dict"

DATASETS_OUT = TRAINING_ROOT / "datasets"

PRETRAIN_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model"
)

# script -> official PP-OCRv5 dict filename, or None to build our own from real labels
OFFICIAL_DICT = {
    "devanagari": "ppocrv5_devanagari_dict.txt",
    "tamil": "ppocrv5_ta_dict.txt",
    "telugu": "ppocrv5_te_dict.txt",
    "urdu": "ppocrv5_arabic_dict.txt",  # Perso-Arabic script family match
}

# script -> official PP-OCRv5 mobile rec pretrained checkpoint filename to
# fine-tune from. Scripts without their own official checkpoint use
# Devanagari's as a generic Indic-abugida backbone initializer.
PRETRAIN_CHECKPOINT = {
    "devanagari": "devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "tamil": "ta_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "telugu": "te_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "urdu": "arabic_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "bengali": "devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "gujarati": "devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "kannada": "devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "malayalam": "devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "odia": "devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams",
    "punjabi": "devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams",
}

# Ol Chiki has zero real data in raw-data/ -- documented gap, not attempted here.
SCRIPTS = list(PRETRAIN_CHECKPOINT.keys())

SEED = 42
VAL_FRACTION = 0.1
VAL_MIN = 20


def load_raw_records(script: str) -> list[tuple[Path, str]]:
    """(image_path, label) pairs from raw-data/<script>/normalized.

    A record file is normally one dict (one full-image word crop = one
    label). A handful of Bengali files are lists of field-level records on a
    shared multi-field page image (IndicDLP form drafts) -- those need a
    bbox crop step this script doesn't do, so they're skipped here (7 of
    ~1,000 Bengali records affected)."""
    norm_dir = RAW_DATA_DIR / script / "normalized"
    pairs = []
    for json_path in sorted(norm_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            continue  # multi-field page image, needs bbox cropping -- skip for now
        gt = (data.get("groundTruth") or "").strip()
        image_filename = data.get("imageFilename")
        if not gt or not image_filename:
            continue
        image_path = norm_dir / image_filename
        if not image_path.exists():
            continue
        pairs.append((image_path, gt))
    return pairs


def load_held_out_filenames(script: str) -> set[str]:
    """Real (non-synthetic) image filenames already used as the eval set in
    demo/rebenchmark_all.py -- must never appear in training data."""
    held_out = set()
    for split in ("train", "val", "test"):
        script_dir = ANNOTATED_DIR / split / script
        if not script_dir.is_dir():
            continue
        for json_path in script_dir.glob("*.json"):
            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                is_synthetic = any(
                    "synthetic" in s.lower() for s in rec.get("collectionSource", [])
                )
                if not is_synthetic:
                    held_out.add(rec["imageFilename"])
    return held_out


def sanitize_label(label: str) -> str:
    return label.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def build_dataset(script: str) -> dict:
    raw_pairs = load_raw_records(script)
    held_out = load_held_out_filenames(script)

    pool = [(p, l) for p, l in raw_pairs if p.name not in held_out]
    excluded_n = len(raw_pairs) - len(pool)

    rng = random.Random(SEED)
    rng.shuffle(pool)
    val_n = max(VAL_MIN, round(len(pool) * VAL_FRACTION))
    val_n = min(val_n, len(pool) // 3)  # never let val eat most of a tiny pool
    val_pairs = pool[:val_n]
    train_pairs = pool[val_n:]

    out_dir = DATASETS_OUT / script
    out_dir.mkdir(parents=True, exist_ok=True)

    dict_name = OFFICIAL_DICT.get(script)
    if dict_name:
        shutil.copyfile(OFFICIAL_DICT_DIR / dict_name, out_dir / "dict.txt")
        dict_source = f"official:{dict_name}"
        dict_size = sum(1 for _ in open(out_dir / "dict.txt", encoding="utf-8"))
    else:
        chars = set()
        for _, label in raw_pairs:  # full pool incl. held-out, so eval chars are covered
            chars.update(c for c in label if c != " ")
        sorted_chars = sorted(chars, key=ord)
        with open(out_dir / "dict.txt", "w", encoding="utf-8") as f:
            for c in sorted_chars:
                f.write(c + "\n")
        dict_source = "custom:real-data-chars"
        dict_size = len(sorted_chars)

    for name, pairs in (("train.txt", train_pairs), ("val.txt", val_pairs)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for image_path, label in pairs:
                f.write(f"{image_path}\t{sanitize_label(label)}\n")

    return {
        "script": script,
        "raw_total": len(raw_pairs),
        "held_out_excluded": excluded_n,
        "train_n": len(train_pairs),
        "val_n": len(val_pairs),
        "dict_source": dict_source,
        "dict_size": dict_size,
        "pretrain": PRETRAIN_CHECKPOINT[script],
    }


def main():
    DATASETS_OUT.mkdir(parents=True, exist_ok=True)
    rows = [build_dataset(script) for script in SCRIPTS]

    header = f"{'script':<12}{'raw':>6}{'excl':>6}{'train':>7}{'val':>6}{'dict':>8}  dict_source / pretrain init"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['script']:<12}{r['raw_total']:>6}{r['held_out_excluded']:>6}"
            f"{r['train_n']:>7}{r['val_n']:>6}{r['dict_size']:>8}  "
            f"{r['dict_source']} / {r['pretrain']}"
        )

    summary_path = TRAINING_ROOT / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
