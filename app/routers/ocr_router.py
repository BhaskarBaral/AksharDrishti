import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.dataset import reader
from app.engines.registry import engines_for_script
from app.imaging import bbox_crop, safe_cer

router = APIRouter(prefix="/api/ocr", tags=["ocr"])


@router.get("/engines/{script}")
def engines_for(script: str):
    return {"script": script, "engines": [e.name for e in engines_for_script(script)]}


def _filter_engines(script: str, requested: list[str] | None):
    """engines_for_script(), optionally narrowed to a caller-picked subset --
    lets the UI run only the engine(s) the user selected instead of all of them."""
    engines = engines_for_script(script)
    if requested:
        wanted = set(requested)
        engines = [e for e in engines if e.name in wanted]
        if not engines:
            raise HTTPException(
                400, f"None of the requested engines {requested!r} support script {script!r}"
            )
    elif not engines:
        raise HTTPException(400, f"No OCR engine has language support for script {script!r}")
    return engines


@router.post("/run-on-sample")
def run_on_sample(script: str = Form(...), split: str = Form(...),
                   filename: str = Form(...), field_index: int = Form(...),
                   engines: list[str] | None = Form(None)):
    """OCR one field of a known dataset sample, scored against its ground truth."""
    try:
        image_path = reader.resolve_image_path(script, split, filename)
        field_records = reader.get_records(script, split, filename)
    except reader.SampleNotFoundError as e:
        raise HTTPException(404, str(e))

    if not (0 <= field_index < len(field_records)):
        raise HTTPException(400, f"field_index out of range (0..{len(field_records) - 1})")

    record = field_records[field_index]
    ground_truth = record["groundTruth"]
    im = Image.open(image_path)
    crop = bbox_crop(im, record["boundingBox"]["vertices"])

    engines = _filter_engines(script, engines)

    results = []
    for engine in engines:
        hyp = engine.recognize(crop, script)
        results.append({
            "engine": engine.name,
            "hypothesis": hyp,
            "cer": safe_cer(ground_truth, hyp),
        })

    return {"groundTruth": ground_truth, "results": results}


@router.post("/run-on-upload")
async def run_on_upload(script: str = Form(...), file: UploadFile = File(...),
                         engines: list[str] | None = Form(None)):
    """Best-effort OCR of a user-uploaded image (whole image, no bounding box,
    no ground truth to score against)."""
    engines = _filter_engines(script, engines)

    contents = await file.read()
    try:
        im = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image")

    results = [{"engine": e.name, "hypothesis": e.recognize(im, script)} for e in engines]
    return {"results": results}


@router.post("/run-page-upload")
async def run_page_upload(script: str = Form(...), file: UploadFile = File(...),
                           engines: list[str] | None = Form(None)):
    """Structured page-level extraction (per-line text + bbox + confidence, in
    reading order) for whichever engines have their own text detector -- see
    OCREngine.supports_page_level. No ground truth to score against."""
    candidates = [e for e in engines_for_script(script) if e.supports_page_level]
    if engines:
        wanted = set(engines)
        candidates = [e for e in candidates if e.name in wanted]
    if not candidates:
        raise HTTPException(
            400, f"No page-level-capable engine available for script {script!r}"
        )

    contents = await file.read()
    try:
        im = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image")

    results = []
    for engine in candidates:
        lines = engine.recognize_page(im, script)
        results.append({
            "engine": engine.name,
            "lines": lines,
            "text": "\n".join(l["text"] for l in lines),
        })
    return {"results": results}
