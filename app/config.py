"""
Central config: filesystem paths this app reads from or writes to.

The dataset repo (Akshar_dristhii_code, Stage 2 deliverable) is a sibling
folder read read-only -- this app never writes into it and never depends on
its code, matching the convention already established by
../ocr-benchmark/benchmark_ocr.py.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

DEMO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(DEMO_ROOT / ".env")
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

# Bhashini/ULCA consumer integration (app/engines/bhashini_ocr_engine.py) --
# credentials from https://bhashini.gov.in/ulca -> My Profile -> API Keys,
# plus the ID of a pipeline you've subscribed to that includes an OCR task.
# Left unset by default: the engine reports itself unavailable rather than
# failing per-request when these are missing.
BHASHINI_USER_ID = os.environ.get("BHASHINI_USER_ID")
BHASHINI_ULCA_API_KEY = os.environ.get("BHASHINI_ULCA_API_KEY")
BHASHINI_PIPELINE_ID = os.environ.get("BHASHINI_PIPELINE_ID")
BHASHINI_CONFIG_URL = os.environ.get(
    "BHASHINI_CONFIG_URL",
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
)
