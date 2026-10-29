import io
import json
import time
import uuid
from functools import lru_cache

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from app.config import TESTDATA_DIR, UPLOADS_DIR
from app.dataset import reader
from app.engines.registry import engines_for_script
from app.inference import INFERENCE_LOCK
from app.imaging import bbox_crop, safe_cer

router = APIRouter(prefix="/api/ocr", tags=["ocr"])

# ULCA language/script codes for every script page-level extraction supports
# -- copied verbatim from the Stage 2 dataset repo's own records
# (synthetic/<script>/kv/clean/*.json "languages" blocks, all inside
# schemas/ocr-record.schema.json's enums), not invented here.
ULCA_LANGUAGE = {
    "devanagari": {"sourceLanguage": "hi", "sourceLanguageName": "Hindi", "sourceScriptCode": "Deva"},
    "bengali": {"sourceLanguage": "bn", "sourceLanguageName": "Bengali", "sourceScriptCode": "Beng"},
    "tamil": {"sourceLanguage": "ta", "sourceLanguageName": "Tamil", "sourceScriptCode": "Taml"},
    "telugu": {"sourceLanguage": "te", "sourceLanguageName": "Telugu", "sourceScriptCode": "Telu"},
    "urdu": {"sourceLanguage": "ur", "sourceLanguageName": "Urdu", "sourceScriptCode": "Arab"},
    "gujarati": {"sourceLanguage": "gu", "sourceLanguageName": "Gujarati", "sourceScriptCode": "Gujr"},
    "kannada": {"sourceLanguage": "kn", "sourceLanguageName": "Kannada", "sourceScriptCode": "Knda"},
    "malayalam": {"sourceLanguage": "ml", "sourceLanguageName": "Malayalam", "sourceScriptCode": "Mlym"},
    "odia": {"sourceLanguage": "or", "sourceLanguageName": "Odia", "sourceScriptCode": "Orya"},
    "punjabi": {"sourceLanguage": "pa", "sourceLanguageName": "Punjabi", "sourceScriptCode": "Guru"},
}

# Page-level engine order per script, best first -- the UI preselects the
# first one that's available and lets the operator switch. Measured, not
# guessed (2026-10-27, page CER via /run-page-upload's ground_truth scoring):
# PRINTED = the dataset repo's synthetic kv pages (2 clean + 1 moderate per
# script), HANDWRITTEN = ICDAR 2024 HWD real pages (Bengali/Telugu, the only
# scripts with any) plus each script's synthetic hw form page.
#   printed mean page CER (tesseract / easyocr / paddleocr / -finetuned):
#   devanagari .033/.050/.011/.446  tamil .079/  -  /.013/.303
#   bengali    .002/.037/  -  /.480  telugu .037/.023/  -  /.487
#   kannada    .042/.037/  -  /.506  urdu   .407/.309/  -  /.840
#   gujarati .033/.507  malayalam .139/.413  odia .059/.594  punjabi .285/.452
#     (those four: tesseract / -finetuned only)
#   real handwritten pages: bengali .799/.722/ - /.545, telugu .806/.810/ - /.721
# Telugu/Kannada: easyocr also wins the synthetic handwriting-font form
# (.086 vs .315, .043 vs .321) at ~1-3.5s a page warm on CPU, so it leads.
# Urdu: easyocr is clearly better but slow on CPU (~17-20s a page).
# Handwritten: on real handwriting (all 10 scripts' real word crops, and the
# Bengali/Telugu real pages) our handwriting-trained fine-tune wins, so it
# leads that order for every script; /page-samples still points the
# synthetic handwriting-font forms at the printed order, which reads them
# far better.
PAGE_ENGINE_PREFERENCE = {
    "devanagari": ["paddleocr", "tesseract", "easyocr", "paddleocr-finetuned"],
    "tamil": ["paddleocr", "tesseract", "paddleocr-finetuned"],
    "bengali": ["tesseract", "easyocr", "paddleocr-finetuned"],
    "telugu": ["easyocr", "tesseract", "paddleocr-finetuned"],
    "kannada": ["easyocr", "tesseract", "paddleocr-finetuned"],
    "urdu": ["easyocr", "tesseract", "paddleocr-finetuned"],
    "gujarati": ["tesseract", "paddleocr-finetuned"],
    "malayalam": ["tesseract", "paddleocr-finetuned"],
    "odia": ["tesseract", "paddleocr-finetuned"],
    "punjabi": ["tesseract", "paddleocr-finetuned"],
}
PAGE_ENGINE_PREFERENCE_HW = {
    script: ["paddleocr-finetuned"] + [e for e in order if e != "paddleocr-finetuned"]
    for script, order in PAGE_ENGINE_PREFERENCE.items()
}

# Very large uploads (12MP phone photos) are downscaled before page OCR --
# no engine here gains accuracy past this, and CPU detection time/memory
# grows with pixel count. Boxes are scaled back to original-image pixels.
PAGE_MAX_SIDE = 2400

ERROR_PREFIX = "__ERROR__:"

# Shared with the /v1 API and Layout routers -- see app/inference.py.
_INFERENCE_LOCK = INFERENCE_LOCK


def _locked(fn, *args, **kwargs):
    with _INFERENCE_LOCK:
        return fn(*args, **kwargs)

IMAGE_FORMAT_MAP = {"JPEG": "jpeg", "PNG": "png", "BMP": "bmp", "TIFF": "tiff"}


@router.get("/engines/{script}")
def engines_for(script: str):
    engines = engines_for_script(script)
    page_capable = [e.name for e in engines if e.supports_page_level]
    return {
        "script": script,
        "engines": [e.name for e in engines],
        # page-capable subset, best first (see PAGE_ENGINE_PREFERENCE)
        "pageEngines": _ranked(page_capable, PAGE_ENGINE_PREFERENCE.get(script, [])),
        "pageEnginesHandwritten": _ranked(page_capable, PAGE_ENGINE_PREFERENCE_HW.get(script, [])),
    }


def _ranked(names: list[str], order: list[str]) -> list[str]:
    return sorted(names, key=lambda n: order.index(n) if n in order else len(order))


def _readable_error(e) -> str:
    """First line of an engine failure, without the internal __ERROR__ marker
    or a multi-KB state_dict/stack dump -- this is what the UI shows."""
    msg = str(e)
    if msg.startswith(ERROR_PREFIX):
        msg = msg[len(ERROR_PREFIX):]
    msg = next((ln.strip() for ln in msg.splitlines() if ln.strip()), "") or "unknown error"
    return msg if len(msg) <= 240 else msg[:237] + "..."


def _word_result(engine_name: str, hyp: str, ground_truth: str | None = None) -> dict:
    """One word-level result card -- an engine failure comes back as a
    readable `error` (and no CER) instead of a raw "__ERROR__:..." hypothesis."""
    out = {"engine": engine_name, "hypothesis": hyp}
    if hyp.startswith(ERROR_PREFIX):
        out = {"engine": engine_name, "hypothesis": "", "error": _readable_error(hyp)}
        if ground_truth is not None:
            out["cer"] = None
    elif ground_truth is not None:
        out["cer"] = safe_cer(ground_truth, hyp)
    return out


def _open_upload(contents: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(contents))
        im.load()
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image")
    # phone photos: the browser shows them EXIF-rotated, so OCR (and the boxes
    # drawn back over that preview) must use the same orientation
    return ImageOps.exif_transpose(im)


def _looks_like_full_page(im: Image.Image) -> bool:
    """Word-level uploads range from one word to a whole page. A tight
    word/line crop is short or very wide; anything else goes down the
    engines' own detection path (is_full_page=True) instead of single-line
    recognition, which would only ever read one line of it."""
    w, h = im.size
    return min(w, h) >= 300 and w / max(h, 1) < 4


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
        try:
            with _INFERENCE_LOCK:
                hyp = engine.recognize(crop, script)
        except Exception as e:
            hyp = f"{ERROR_PREFIX}{e}"
        results.append(_word_result(engine.name, hyp, ground_truth))

    return {"groundTruth": ground_truth, "results": results}


@router.post("/run-on-upload")
async def run_on_upload(script: str = Form(...), file: UploadFile = File(...),
                         engines: list[str] | None = Form(None)):
    """Best-effort OCR of a user-uploaded image (whole image, no bounding box,
    no ground truth to score against)."""
    engines = _filter_engines(script, engines)

    im = _open_upload(await file.read())
    full_page = _looks_like_full_page(im)

    results = []
    for engine in engines:
        try:
            # async endpoint: run off the event loop so a long page doesn't
            # freeze every other request (engine lists, images) meanwhile
            hyp = await run_in_threadpool(_locked, engine.recognize, im, script, is_full_page=full_page)
        except Exception as e:
            hyp = f"{ERROR_PREFIX}{e}"
        results.append(_word_result(engine.name, hyp))
    return {"results": results, "treatedAsFullPage": full_page}


@router.post("/benchmark")
def benchmark(script: str = Form(...), engines: list[str] | None = Form(None),
              limit: int = Form(15)):
    """Aggregate CER per engine across up to `limit` real dataset samples --
    the same head-to-head comparison rebenchmark_all.py runs offline,
    exposed live so the UI can score a remote engine (bhashini-ocr) against
    the local ones on identical inputs. Each sample is scored on its field
    with the longest ground truth, same heuristic the Playground tab uses
    when auto-picking a field."""
    engines = _filter_engines(script, engines)
    limit = max(1, min(limit, 40))
    samples = reader.list_samples(script)[:limit]

    per_engine = {e.name: [] for e in engines}
    per_sample = []

    for s in samples:
        try:
            records = reader.get_records(script, s["split"], s["filename"])
            image_path = reader.resolve_image_path(script, s["split"], s["filename"])
        except reader.SampleNotFoundError:
            continue
        if not records:
            continue

        best = max(records, key=lambda r: len(r.get("groundTruth") or ""))
        ground_truth = best["groundTruth"]
        im = Image.open(image_path)
        crop = bbox_crop(im, best["boundingBox"]["vertices"])

        row = {"split": s["split"], "filename": s["filename"], "groundTruth": ground_truth, "results": {}}
        for engine in engines:
            with _INFERENCE_LOCK:
                hyp = engine.recognize(crop, script)
            cer = safe_cer(ground_truth, hyp)
            if cer is not None:
                per_engine[engine.name].append(cer)
            row["results"][engine.name] = {"hypothesis": hyp, "cer": cer}
        per_sample.append(row)

    summary = [
        {
            "engine": e.name,
            "meanCer": round(sum(per_engine[e.name]) / len(per_engine[e.name]), 4) if per_engine[e.name] else None,
            "scored": len(per_engine[e.name]),
            "total": len(per_sample),
        }
        for e in engines
    ]
    summary.sort(key=lambda r: (r["meanCer"] is None, r["meanCer"]))

    return {"script": script, "sampleCount": len(per_sample), "summary": summary, "samples": per_sample}


@router.post("/run-page-upload")
async def run_page_upload(script: str = Form(...), file: UploadFile = File(...),
                           engines: list[str] | None = Form(None),
                           ground_truth: str | None = Form(None)):
    """Structured page-level extraction (per-line text + bbox + confidence, in
    reading order) for whichever engines have their own text detector -- see
    OCREngine.supports_page_level. When the page is a known dataset sample
    (see /page-samples) the caller passes its page `ground_truth` and each
    engine also gets a page CER (whitespace-normalised, in the reading order
    the engine returned)."""
    candidates = [e for e in engines_for_script(script) if e.supports_page_level]
    if engines:
        wanted = set(engines)
        candidates = [e for e in candidates if e.name in wanted]
    if not candidates:
        raise HTTPException(
            400, f"No page-level-capable engine available for script {script!r}"
        )

    im = _open_upload(await file.read())
    scale = 1.0
    if max(im.size) > PAGE_MAX_SIDE:
        scale = max(im.size) / PAGE_MAX_SIDE
        im = im.resize((round(im.width / scale), round(im.height / scale)), Image.LANCZOS)

    gt_norm = " ".join(ground_truth.split()) if ground_truth else None

    results = []
    for engine in candidates:
        started = time.time()
        try:
            lines, error = await run_in_threadpool(_locked, engine.recognize_page, im, script), None
        except Exception as e:
            lines, error = [], _readable_error(e)
        for l in lines:
            l["bbox"] = [[float(x) * scale, float(y) * scale] for x, y in l["bbox"]]
        result = {
            "engine": engine.name,
            "lines": lines,
            "text": "\n".join(l["text"] for l in lines),
            "seconds": round(time.time() - started, 2),
        }
        if error:
            result["error"] = error
        if gt_norm:
            result["cer"] = None if error else safe_cer(gt_norm, " ".join(result["text"].split()))
        results.append(result)
    return {"results": results, "groundTruth": gt_norm}


def _page_ground_truth(entry: dict) -> str:
    """Synthetic pages carry one line per form field -> rebuild the page text
    top-to-bottom; real pages are already one whole-page transcription."""
    lines = entry.get("lines")
    if not lines:
        return " ".join(entry["groundTruth"].split())

    def top_left(l):
        v = (l.get("boundingBox") or {}).get("vertices") or [{"x": 0, "y": 0}]
        return (min(p["y"] for p in v), min(p["x"] for p in v))

    return " ".join((l.get("groundTruth") or "").strip() for l in sorted(lines, key=top_left)).strip()


@lru_cache(maxsize=32)
def _page_samples(script: str) -> tuple:
    """Page level's sample strip: TestData/<script>/page_level/ (handwritten
    pages, held out -- real where a real source exists, else synthetic
    handwriting-style forms; see TestData/README.md)."""
    manifest = TESTDATA_DIR / script / "page_level" / "manifest.json"
    if not manifest.exists():
        return ()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    samples = []
    counters = {}
    for e in entries:
        label = "Real handwritten page" if e["source"] == "real" else "Handwritten form (synthetic)"
        counters[label] = counters.get(label, 0) + 1
        samples.append({
            "id": f"{script}/page_level/{e['file']}",
            "filename": e["file"],
            "label": f"{label} {counters[label]}",
            "imageTextType": "handwritten-text",
            "source": e["source"],
            "groundTruth": _page_ground_truth(e),
        })
    return tuple(samples)


@router.get("/page-samples/{script}")
def page_samples(script: str):
    """A few full-page images per script from the dataset repo with their
    page-level ground truth, so Page level can be demoed and scored without
    hunting for files. Each sample carries the engine to preselect for it:
    real handwritten pages get the handwriting order, everything else
    (including the synthetic "handwritten" forms, which are rendered in
    handwriting-style fonts and read far better with the printed engines --
    page CER ~0.02-0.3 vs ~0.5-0.65 for the fine-tune) the printed order."""
    page_capable = [e.name for e in engines_for_script(script) if e.supports_page_level]
    printed = _ranked(page_capable, PAGE_ENGINE_PREFERENCE.get(script, []))
    handwritten = _ranked(page_capable, PAGE_ENGINE_PREFERENCE_HW.get(script, []))
    samples = []
    for s in _page_samples(script):
        order = handwritten if s["source"] == "real" else printed
        samples.append({**s, "recommendedEngine": order[0] if order else None})
    return {"script": script, "samples": samples}


@router.get("/page-samples/{script}/image")
def page_sample_image(script: str, id: str):
    if id not in {s["id"] for s in _page_samples(script)}:
        raise HTTPException(404, f"Unknown page sample {id!r} for {script!r}")
    return FileResponse(TESTDATA_DIR / id)


@router.post("/create-ground-truth")
async def create_ground_truth(
    script: str = Form(...),
    file: UploadFile = File(...),
    lines: str = Form(...),
    engine: str = Form(...),
    image_text_type: str = Form("computer-typed-text"),
):
    """Turn a reviewed Document-mode extraction into real ULCA-schema records
    and save them to disk -- the actual point of the "Create Ground Truth &
    Record" button, not another mock screen. `lines` is a JSON array of
    {bbox: [[x,y],...], text, confidence} as edited by the operator in
    DocumentMode.jsx (post-correction text, not necessarily the raw OCR
    output). One ULCA record is written per non-empty line, same field names
    as the Stage 2 dataset repo's schema -- this app never writes into that
    repo (see app/config.py), so the save lands in this app's own uploads/
    folder instead.
    """
    if script not in ULCA_LANGUAGE:
        raise HTTPException(400, f"No ULCA language/script mapping for {script!r} yet")
    if image_text_type not in ("computer-typed-text", "handwritten-text", "scene-text", "typewriter-typed-text"):
        raise HTTPException(400, f"Invalid image_text_type: {image_text_type!r}")

    try:
        parsed_lines = json.loads(lines)
        if not isinstance(parsed_lines, list):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "`lines` must be a JSON array")

    contents = await file.read()
    try:
        im = Image.open(io.BytesIO(contents))
        im.load()
    except Exception:
        raise HTTPException(400, "Could not read uploaded file as an image")
    fmt = IMAGE_FORMAT_MAP.get(im.format, "png")

    records = []
    for line in parsed_lines:
        text = (line.get("text") or "").strip()
        bbox = line.get("bbox")
        if not text or not bbox:
            continue
        vertices = [{"x": round(float(pt[0]), 2), "y": round(float(pt[1]), 2)} for pt in bbox]
        records.append({
            "datasetType": "ocr-corpus",
            "languages": dict(ULCA_LANGUAGE[script]),
            "collectionSource": [f"citizen upload -- {engine} OCR, operator-reviewed in AksharDrishti demo"],
            "domain": ["general"],
            "license": "unknown-license",
            "submitter": {
                "name": "AksharDrishti citizen upload",
                "aboutMe": "BHASHINI AksharDrishti Hackathon Stage 3 demo",
            },
            "format": fmt,
            "dpi": "72_dpi",
            "imageTextType": image_text_type,
            "imageFilename": None,  # filled in once the filename is known, below
            "groundTruth": text,
            "boundingBox": {"vertices": vertices},
        })

    if not records:
        raise HTTPException(400, "No non-empty lines with a bounding box to save")

    submission_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    out_dir = UPLOADS_DIR / script / submission_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image_filename = f"{submission_id}.{fmt}"
    with open(out_dir / image_filename, "wb") as f:
        f.write(contents)
    for record in records:
        record["imageFilename"] = image_filename

    json_path = out_dir / f"{submission_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return {
        "submissionId": submission_id,
        "savedTo": str(out_dir),
        "imageFilename": image_filename,
        "recordCount": len(records),
        "records": records,
    }
