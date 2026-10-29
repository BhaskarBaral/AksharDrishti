import io
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image

from app.config import TESTDATA_DIR
from app.inference import INFERENCE_LOCK
from app.layout_detector import detector
from app.layout_preserve import analyze

router = APIRouter(prefix="/api/layout", tags=["layout"])

LAYOUT_TESTDATA = TESTDATA_DIR / "layout"
# TestData/layout/<group>/ (built by TestData/_tools/build_testdata.py), in
# the order the Layout tab lists them.
SAMPLE_GROUPS = [
    ("real_labelled", "Real pages with answer key (IndicDLP Odia test split)"),
    ("real_unlabelled", "Real complex forms (no answer key)"),
    ("synthetic_labelled", "Synthetic forms with seals / stamps / signatures"),
]


def _build_samples() -> list[dict]:
    samples = []
    for group, _ in SAMPLE_GROUPS:
        manifest = LAYOUT_TESTDATA / group / "manifest.json"
        if not manifest.exists():
            continue
        for n, e in enumerate(json.loads(manifest.read_text(encoding="utf-8")), 1):
            if group == "synthetic_labelled":
                label = f"{e['script']} · {e['doctype']} · {e['quality']}"
            elif group == "real_labelled":
                label = f"Odia page {n}"
            else:
                label = e["file"].rsplit(".", 1)[0]
            samples.append({"id": f"{group}/{e['file']}", "group": group, "label": label, "script": e["script"],
                            "path": LAYOUT_TESTDATA / group / e["file"], "groundTruth": e.get("regions")})
    return samples


_samples_cache: list[dict] | None = None


def _samples() -> list[dict]:
    global _samples_cache
    if _samples_cache is None:
        _samples_cache = _build_samples()
    return _samples_cache


def _sample(sample_id: str) -> dict:
    s = next((s for s in _samples() if s["id"] == sample_id), None)
    if s is None:
        raise HTTPException(404, "Unknown sample")
    return s


@router.get("/status")
def status():
    return {"available": detector.is_available(), "model": detector.model_name}


@router.get("/samples")
def samples():
    return {
        "groups": [{"id": g, "label": label} for g, label in SAMPLE_GROUPS],
        "samples": [{k: s[k] for k in ("id", "group", "label", "script")} for s in _samples()],
    }


@router.get("/samples/image")
def sample_image(id: str):
    return FileResponse(_sample(id)["path"])


@router.post("/analyze-sample")
def analyze_sample(id: str = Form(...), script: str | None = Form(None), threshold: float = Form(0.3)):
    s = _sample(id)
    result = _analyze(Image.open(s["path"]), script or s["script"], threshold)
    result["groundTruth"] = s["groundTruth"]
    return result


@router.post("/analyze-upload")
async def analyze_upload(file: UploadFile = File(...), script: str = Form(...), threshold: float = Form(0.3)):
    try:
        im = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image")
    result = await run_in_threadpool(_analyze, im, script, threshold)
    result["groundTruth"] = None
    return result


def _analyze(im: Image.Image, script: str, threshold: float) -> dict:
    if not detector.is_available():
        raise HTTPException(503, "Layout model weights not found")
    with INFERENCE_LOCK:
        return analyze(im, script, threshold)
