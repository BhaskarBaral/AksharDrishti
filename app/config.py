"""
Central config: filesystem paths this app reads from or writes to.

The dataset repo (Akshar_dristhii_code, Stage 2 deliverable) is a sibling
folder read read-only -- this app never writes into it and never depends on
its code, matching the convention already established by
../ocr-benchmark/benchmark_ocr.py.
"""
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO_ROOT = DEMO_ROOT.parent

DATASET_REPO = MODEL_REPO_ROOT.parent / "Akshar_dristhii_code"
ANNOTATED_DIR = DATASET_REPO / "annotated"
ANNOTATED_SPLITS = ["test", "val", "train"]

TESSDATA_DIR = MODEL_REPO_ROOT / "ocr-benchmark" / "tessdata"
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

STATIC_DIR = DEMO_ROOT / "static"

CROP_PAD_PX = 8

# Dataset browser: most scripts have far more synthetic (template-generated)
# samples than real ones, which buries the real-world samples that actually
# matter for judging engine accuracy. Cap synthetic samples shown per script;
# real samples (anything whose collectionSource isn't "synthetic..." /
# "synthetic-mock...") are never capped.
SYNTHETIC_SAMPLE_CAP = 4
