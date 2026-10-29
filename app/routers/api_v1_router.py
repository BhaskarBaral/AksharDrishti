"""Public, versioned API (/v1) -- the clean integrator-facing surface behind
the API playground tab. Thin wrapper over the same engines and layout
detector the /api/* demo routes use; responses are ULCA OCR records (see
app/ulca.py) or plain text, and every failure is a real HTTP error code with
a {"detail": ...} body, never a 200 carrying an __ERROR__ string."""
import io
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app import ulca
from app.config import DATASET_REPO, DEMO_ROOT
from app.engines.registry import engines_for_script
from app.inference import INFERENCE_LOCK
from app.layout_detector import detector
from app.routers.ocr_router import PAGE_ENGINE_PREFERENCE

router = APIRouter(prefix="/v1", tags=["public-api"])

API_VERSION = "1.0"
SCRIPTS = list(ulca.ULCA_LANGUAGE)
LEVELS = ("word", "page")
FORMATS = ("ulca", "text")
# Default engine order when the caller doesn't pick one (first available wins;
# if it fails at runtime the next is tried). Picked by measured CER on this
# app's own test images, not by assumption:
#  - word: our fine-tuned checkpoints beat stock PP-OCRv5 by a wide margin on
#    test_data/ real word crops (e.g. Tamil 0.04 vs 0.83, Devanagari 0.12 vs
#    0.51 mean CER -- caveat: some of those crops are from the training split).
#  - page: per script, the OCR tab's measured printed-page ranking
#    (ocr_router.PAGE_ENGINE_PREFERENCE) -- stock paddleocr for Devanagari/
#    Tamil now that its detector is capped, else Tesseract or EasyOCR.
ENGINE_PREFERENCE = {
    "word": ("paddleocr-finetuned", "paddleocr"),
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Engine pipelines are shared, stateful model objects (and PaddleOCR is not
# thread-safe), while sync endpoints run on FastAPI's threadpool -- serialize
# inference rather than risk two requests inside one predictor at once.
_inference_lock = INFERENCE_LOCK  # app-wide, see app/inference.py


# ---------- helpers ----------

def _check_script(script: str | None) -> str:
    if not script:
        raise HTTPException(400, f"`script` is required -- one of: {', '.join(SCRIPTS)}")
    script = script.strip().lower()
    if script not in ulca.ULCA_LANGUAGE:
        raise HTTPException(400, f"Unsupported script {script!r} -- one of: {', '.join(SCRIPTS)}")
    return script


def _read_image(file: UploadFile | None) -> tuple[Image.Image, str]:
    if file is None:
        raise HTTPException(400, "`file` is required (multipart image upload)")
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(400, "Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        im = Image.open(io.BytesIO(contents))
        im.load()
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image (PNG, JPEG, BMP or TIFF)")
    name = Path(file.filename or "").name or f"upload.{ulca.image_format(im)}"
    return im, name


def _engines_for_level(script: str, level: str):
    engines = engines_for_script(script)
    if level == "page":
        engines = [e for e in engines if e.supports_page_level]
    return engines


def _candidate_engines(script: str, level: str, requested: str | None) -> list:
    """The engine(s) to try, in order: just the requested one if the caller
    named an engine, else the default preference list (with fallback)."""
    available = _engines_for_level(script, level)
    if requested:
        requested = requested.strip()
        match = next((e for e in available if e.name == requested), None)
        if match:
            return [match]
        all_for_script = {e.name for e in engines_for_script(script)}
        if requested in all_for_script:
            raise HTTPException(400, f"Engine {requested!r} does not support level={level!r}")
        names = ", ".join(e.name for e in available) or "none"
        raise HTTPException(400, f"Engine {requested!r} is not available for script {script!r} "
                                 f"at level={level!r} (available: {names})")
    if not available:
        raise HTTPException(503, f"No OCR model is currently available for script {script!r} at level={level!r}")
    order = PAGE_ENGINE_PREFERENCE.get(script, []) if level == "page" else ENGINE_PREFERENCE[level]
    preferred = [e for name in order for e in available if e.name == name]
    return preferred or available[:1]


def _short(e: Exception) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return msg[:300]


def _run_word(engine, im: Image.Image, script: str) -> str:
    with _inference_lock:
        try:
            text = engine.recognize(im, script)
        except Exception as e:
            raise HTTPException(503, f"Engine {engine.name!r} failed: {_short(e)}")
    if text.startswith("__ERROR__"):
        raise HTTPException(503, f"Engine {engine.name!r} failed: {text[len('__ERROR__:'):][:300]}")
    return text.strip()


def _run_page(engine, im: Image.Image, script: str) -> list[dict]:
    with _inference_lock:
        try:
            lines = engine.recognize_page(im, script)
        except Exception as e:
            raise HTTPException(503, f"Engine {engine.name!r} failed: {_short(e)}")
    return [
        {
            "text": (l.get("text") or "").strip(),
            "bbox": [[round(float(x), 2), round(float(y), 2)] for x, y in l["bbox"]],
            "confidence": round(float(l["confidence"]), 4) if l.get("confidence") is not None else None,
        }
        for l in lines
        if (l.get("text") or "").strip()
    ]


# ---------- endpoints ----------

@router.get("/health")
def health():
    return {
        "status": "ok",
        "version": API_VERSION,
        "ocr": {"scripts": [s for s in SCRIPTS if engines_for_script(s)]},
        "layout": {"available": detector.is_available()},
    }


@router.get("/scripts")
def scripts():
    out = []
    for script in SCRIPTS:
        engines = engines_for_script(script)
        word_default = _default_name(script, "word")
        page_default = _default_name(script, "page")
        out.append({
            "script": script,
            **ulca.ULCA_LANGUAGE[script],
            "levels": [lvl for lvl in LEVELS if _engines_for_level(script, lvl)],
            "engines": [
                {"name": e.name, "levels": ["word", "page"] if e.supports_page_level else ["word"]}
                for e in engines
            ],
            "defaultEngine": {"word": word_default, "page": page_default},
        })
    return {"scripts": out, "formats": list(FORMATS)}


def _default_name(script: str, level: str) -> str | None:
    try:
        return _candidate_engines(script, level, None)[0].name
    except HTTPException:
        return None


@router.post("/ocr/recognize")
def recognize(
    file: UploadFile | None = File(None),
    script: str | None = Form(None),
    level: str = Form("word"),
    engine: str | None = Form(None),
    format: str = Form("ulca"),
    image_text_type: str = Form("computer-typed-text"),
):
    """OCR one image. level=word treats the whole image as a single cropped
    word/line; level=page detects and recognizes every text line. format=ulca
    returns ULCA OCR records (one per recognized line), format=text returns
    the transcript (plus per-line boxes at page level)."""
    t0 = time.perf_counter()
    script = _check_script(script)
    level = (level or "word").strip().lower()
    if level not in LEVELS:
        raise HTTPException(400, f"Invalid level {level!r} -- one of: {', '.join(LEVELS)}")
    fmt_out = (format or "ulca").strip().lower()
    if fmt_out not in FORMATS:
        raise HTTPException(400, f"Invalid format {format!r} -- one of: {', '.join(FORMATS)}")
    if image_text_type not in ulca.IMAGE_TEXT_TYPES:
        raise HTTPException(400, f"Invalid image_text_type {image_text_type!r} -- one of: "
                                 f"{', '.join(ulca.IMAGE_TEXT_TYPES)}")
    im, filename = _read_image(file)
    candidates = _candidate_engines(script, level, engine)

    # Only the default (unrequested) path has more than one candidate: if
    # that engine fails at runtime, fall back to the next instead of erroring.
    failed = []
    for eng in candidates:
        try:
            if level == "word":
                text = _run_word(eng, im, script)
                lines = [{"text": text, "bbox": [[v["x"], v["y"]] for v in ulca.full_image_vertices(im)],
                          "confidence": None}] if text else []
            else:
                lines = _run_page(eng, im, script)
                text = "\n".join(l["text"] for l in lines)
            break
        except HTTPException as e:
            if e.status_code != 503 or eng is candidates[-1]:
                raise
            failed.append(eng.name)

    response = {
        "script": script,
        "language": ulca.ULCA_LANGUAGE[script]["sourceLanguage"],
        "level": level,
        "engine": eng.name,
        "format": fmt_out,
        "image": {"filename": filename, "width": im.width, "height": im.height},
    }
    if failed:
        response["fallbackFrom"] = failed
    if fmt_out == "text":
        response["text"] = text
        if level == "page":
            response["lines"] = lines
    else:
        fmt, dpi = ulca.image_format(im), ulca.image_dpi(im)
        source = f"AksharDrishti API /v1/ocr/recognize -- {eng.name} OCR, {level}-level, unreviewed machine output"
        response["records"] = [
            ulca.build_record(script, l["text"], ulca.quad_vertices(l["bbox"]), fmt=fmt, dpi=dpi,
                              image_filename=filename, image_text_type=image_text_type,
                              collection_source=source)
            for l in lines
        ]
        # Confidence has no field in the ULCA schema (additionalProperties:
        # false), so it's returned alongside, index-aligned with `records`.
        response["confidences"] = [l["confidence"] for l in lines]
    response["recordCount"] = len(lines)
    response["timingMs"] = round((time.perf_counter() - t0) * 1000)
    return response


@router.post("/layout/detect")
def layout_detect(file: UploadFile | None = File(None), threshold: str = Form("0.5")):
    """Page-layout regions (header / form-field / table / seal / stamp /
    signature ...) with pixel bboxes [x0, y0, x1, y1] and scores."""
    t0 = time.perf_counter()
    try:
        threshold = float(threshold)
    except ValueError:
        threshold = -1.0
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(400, "`threshold` must be a number between 0 and 1")
    im, filename = _read_image(file)
    if not detector.is_available():
        raise HTTPException(503, "Layout model is not available on this server")
    with _inference_lock:
        try:
            regions = detector.detect(im, threshold)
        except Exception as e:
            raise HTTPException(503, f"Layout model failed: {_short(e)}")
    return {
        "image": {"filename": filename, "width": im.width, "height": im.height},
        "threshold": threshold,
        "regionCount": len(regions),
        "regions": regions,
        "timingMs": round((time.perf_counter() - t0) * 1000),
    }


@router.get("/samples/{script}")
def sample(script: str, level: str = "page"):
    """One bundled sample image per script, so the playground (and curl users)
    can try the API without hunting for a file: a clean synthetic key-value
    form page for level=page, a real handwritten word crop for level=word."""
    script = _check_script(script)
    if level == "page":
        candidates = sorted((DATASET_REPO / "synthetic" / script / "kv" / "clean").glob("*.png"))
    elif level == "word":
        folder = DEMO_ROOT / "test_data" / script
        candidates = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    else:
        raise HTTPException(400, f"Invalid level {level!r} -- one of: {', '.join(LEVELS)}")
    if not candidates:
        raise HTTPException(404, f"No bundled {level}-level sample for script {script!r}")
    return FileResponse(candidates[0], filename=candidates[0].name, content_disposition_type="inline")
