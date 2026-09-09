# AksharDrishti — Stage 3 Demo

See [CHARTER.md](CHARTER.md) for the problem statement, knowledge base, project
lifecycle, tech stack, and progress log/next milestones.

A working prototype: browse the Stage 2 annotated dataset and run OCR
baselines (Tesseract, EasyOCR, PaddleOCR, TrOCR) against it, or against an
uploaded image, picking which engine(s) to run per request. This is a
serving/demo layer over the baselines already established in
`../ocr-benchmark/` — no new model training here.

EasyOCR and TrOCR run on GPU (`setup_gpu.ps1` installs a CUDA-enabled torch).
PaddleOCR attempts GPU too, but falls back to CPU automatically if
`paddlepaddle-gpu` fails to load -- on this dev machine its pip-installed
cuDNN DLLs hit a Windows `WinError 127` at import time (missing MSVC
redistributable dependency chain), so it currently runs on CPU here. Separately,
`paddlepaddle` must stay pinned to `3.2.2`: 3.3.0/3.3.1 have a CPU (oneDNN/PIR)
regression that breaks every `predict()` call outright. See
`app/engines/paddleocr_engine.py`.

Reads the sibling dataset repo (`../../Akshar_dristhii_code`) **read-only**;
this project never writes into it and never depends on its code, same
convention as `../ocr-benchmark/benchmark_ocr.py`.

## Layout (modular by concern)

```
demo/
  app/
    config.py            paths: dataset repo, tessdata dir, tesseract exe
    imaging.py            bbox cropping + CER scoring, shared by ocr_router
    dataset/
      reader.py           read-only access to annotated/<split>/<script>/*.json+images
    engines/
      base.py              OCREngine interface every backend implements
      tesseract_engine.py  Tesseract backend (script -> tesseract lang code)
      easyocr_engine.py    EasyOCR backend (script -> easyocr lang code, lazy-loaded readers)
      paddleocr_engine.py  PaddleOCR backend (Devanagari + Tamil only, GPU with CPU fallback)
      trocr_engine.py      TrOCR backend (Devanagari handwriting, community checkpoint)
      registry.py          engines_for_script() -- add a new backend here only
    routers/
      dataset_router.py   GET  /api/dataset/... (scripts, samples, records, image)
      ocr_router.py        POST /api/ocr/run-on-sample, /api/ocr/run-on-upload (both take an
                            optional repeated `engines` form field to run a chosen subset)
    main.py                 FastAPI app, wires routers + serves static/
  static/                  vanilla JS single-page UI (dataset browser + OCR runner)
  run.py                    entrypoint: python run.py
  requirements.txt
```

Adding a new OCR backend (e.g. a fine-tuned model later) means writing one
`OCREngine` subclass and registering it in `engines/registry.py` --
`dataset/` and `routers/` don't change.

## Run

Use this folder's own venv (or reuse `../ocr-benchmark/.venv`, which
already has pytesseract/easyocr/jiwer/pillow installed):

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run.py
```

Then open http://127.0.0.1:8000

## What it shows

1. **Dataset** — pick a script, pick a sample image from the annotated test/val/train split.
2. **Sample** — the image with every field's bounding box drawn on it; click a box or a table row.
3. **OCR result** — check which engine(s) to run (all available ones are pre-checked, uncheck to run a subset), then click a field: each selected engine's hypothesis, scored (CER) against the ground truth from the ULCA record.
4. **Try your own image** — upload any image, pick a script and engine(s), get each selected engine's whole-image OCR output (no ground truth to score against).

## Known gaps (carried over from ocr-benchmark's findings)

- Ol Chiki has no engine support at all -- a documented gap, not a bug.
- Gujarati, Malayalam, Odia, Punjabi have Tesseract support only (no deep-learning engine).
- PaddleOCR currently only covers Devanagari and Tamil -- PP-OCRv5 only ships an
  *official* recognition model for those two Indic scripts today. Not extended to
  other scripts speculatively; see `app/engines/paddleocr_engine.py`.
- TrOCR currently only covers Devanagari, via a community fine-tune
  (`aayushpuri01/TrOCR-Devanagari`) trained on handwritten Devanagari/Nepali data --
  the model card states no license, which is a real risk to flag before relying on
  it beyond a demo. No official Microsoft Indic checkpoint exists to use instead.
- Baseline accuracy is low outside Devanagari/Bengali/Tamil (see `../ocr-benchmark/results/summary.md`) -- this demo surfaces that honestly via CER, it doesn't hide it.
- Tesseract, EasyOCR, and PaddleOCR are print-text only; TrOCR is the sole
  handwriting-capable engine today, and only for one script.
