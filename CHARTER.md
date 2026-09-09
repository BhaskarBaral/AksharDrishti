# AksharDrishti Charter

BHASHINI AksharDrishti hackathon — problem statement, knowledge base, project
lifecycle, tech stack, and a running progress log. Companion to
[README.md](README.md); the polished web version (same content, better
presentation) lives at the published Artifact link shared in-conversation —
this file is the git-tracked, always-in-the-repo copy.

_Last updated: 2026-09-08_

---

## 0. Progress log

### Completed this session (2026-09-08)
- [x] **Fresh full 4-engine re-benchmark** — `demo/rebenchmark_all.py` runs the exact production engine classes (not a reimplementation) against every field in `annotated/` (all 3 splits, 1,275 engine invocations), scored separately for real vs. synthetic fields. Raw rows and summary in `demo/results/rebenchmark_all_raw.json` / `rebenchmark_all_summary.md`.
- [x] **Fine-tuning target decided: PaddleOCR** (see Key findings #3 below) — resolves the "not yet decided" item from the previous session.

### Completed previous session
- [x] **PaddleOCR engine added** — official PP-OCRv5 Devanagari + Tamil recognition models, wired into the existing `OCREngine` plugin architecture.
- [x] **GPU acceleration enabled** — CUDA `torch` drives EasyOCR and TrOCR; PaddleOCR attempts GPU and falls back to CPU cleanly (a Windows cuDNN DLL issue on this machine, not a code bug).
- [x] **Two real environment bugs fixed** — PaddlePaddle 3.3.x's CPU inference regression (pinned to `3.2.2`), and a missing `paddlepaddle` dependency that would've silently broken a fresh install on another machine.
- [x] **TrOCR engine added** — the only handwriting-capable engine of the four; also fixed a legacy-tokenizer incompatibility with current `transformers` (use `RobertaTokenizer` explicitly, not `AutoTokenizer`).
- [x] **Manual engine-selection UI built** — checkboxes to run any subset of the 4 engines, in both the dataset browser and the "try your own image" upload panel.
- [x] **Synthetic-sample cap added** to the dataset browser — real samples are never capped, synthetic capped at 4 per script, and visibly tagged rather than silently filtered.
- [x] **TrOCR rigorously stress-tested** — a 57-field benchmark, a real-only re-run, and a fair single-line out-of-distribution test (a random calligraphy photo) — rather than trusting its first promising result.

### Key findings
1. **TrOCR's "perfect" handwriting score was very likely data leakage, not skill.** Its one clean sweep came on samples sourced from IIIT-HW-Dev — almost certainly the same corpus the checkpoint was fine-tuned on. On two independent genuinely-unseen tests (synthetic-mock style, and a real calligraphy photo neither this dataset nor the checkpoint could have seen), TrOCR was the *worst* of the four engines. EasyOCR and PaddleOCR are the two engines actually worth trusting for real handwriting today.
2. **The demo's dataset browser was exposing ~0.03% of the real data that exists.** The curated `annotated/` folder has 15 real + 90 synthetic fields per script — but the sibling repo's `raw-data/` and `synthetic/` folders, never wired into this app, hold **10,943 real** and **414,000 synthetic** labeled fields across 10 scripts. This reversed an earlier (wrong) conclusion that there wasn't enough real data to responsibly fine-tune a model.
3. **TrOCR leakage reconfirmed, more starkly, on the full re-run.** CER 0.0 across all 15 real Devanagari fields, but 0.8061 (near-total failure) on the 90 *synthetic* Devanagari fields. Every other engine finds synthetic (clean, machine-rendered) text easier than real photographed samples — Tesseract 1.06→0.06, EasyOCR 0.77→0.14, PaddleOCR 0.57→0.05, real→synthetic CER. TrOCR is the only one that inverts this, which is the signature of memorizing the specific real images rather than reading them. Not a recommended engine for anything beyond a benchmarked/flagged option in the UI.
4. **Fine-tuning target decided: PaddleOCR.** On both scripts it's tested on, PaddleOCR beats EasyOCR and Tesseract head-to-head on real fields — Devanagari CER 0.5652 vs. 0.7719 (EasyOCR) vs. 1.0563 (Tesseract); Tamil CER 0.8065 vs. 1.0 (EasyOCR) vs. 0.9831 (Tesseract) — and by a wider margin on synthetic fields. It also ships official PaddleX training tooling matched to its own architecture, unlike EasyOCR (no first-class fine-tuning path). This reverses the earlier lean toward EasyOCR from a smaller, less controlled test. **The catch:** official PaddleOCR recognition coverage today is only Devanagari + Tamil — the fine-tuning roadmap must *extend* it to the other 8 scripts (Bengali, Urdu, Kannada, Telugu, Gujarati, Malayalam, Odia, Punjabi, Olchiki) using the untapped `raw-data`/`synthetic` pool, not just improve the two scripts it already covers.
5. **None of the four engines are anywhere near production-ready as-is.** Real-field CER is 0.6–1.0+ for nearly every script/engine pair (a CER above 1.0 means the hypothesis has more errors than the ground truth has characters). The honest conclusion of this benchmark isn't "engine X is good" — it's "everything needs fine-tuning on real data before this is deployable," which is exactly what Stage 4 is for.

### Next milestones
1. ~~Decide the fine-tuning target~~ ✅ **Decided: PaddleOCR** (Key finding #4).
2. **Build a real train/val/test split** from the newly-found `raw-data` + `synthetic` pool, separate from the small curated set the demo browses today — so `val`/`test` finally carry real-world signal.
3. **Stand up PaddleX training tooling**, prioritizing the 8 scripts PaddleOCR has zero official recognition coverage for today, and an honest eval loop reusing the existing `safe_cer` scoring.
4. **Run the fine-tune and report an honest held-out number** — resisting the same trap TrOCR fell into: no declaring victory without a genuinely unseen test.
5. **Longer-term (Stage 4–6, unchanged)** — layout/table detection, post-OCR structured output, Bhashini deployment integration.

---

## 1. Problem statement

Indian government, legal, and citizen-service records exist in wildly
inconsistent formats: complex multi-column layouts, low-resolution scans,
handwritten entries, and text that switches script and language mid-document.
General-purpose OCR degrades badly on exactly this kind of document, which
stalls large-scale digitization, search, and any downstream language
processing.

The mandate is to build OCR that is **robust** to real-world scan quality,
**scalable** across scripts and document types, and **deployment-ready** for
Bhashini's national digital infrastructure — not a lab benchmark result.

**Focus areas:**
- Complex layouts — multi-column text, nested tables, fixed-field forms
- Noisy scans — blur, fading, folds, stamps/seals, skew, low contrast
- Handwriting — full-page cursive, mixed typed/handwritten, regional styles
- Multilingual & code-mixed — script-switching mid-page, language-ID
- Model adaptation — LayoutLM, Donut, DocTR, GPU fine-tuning
- Post-OCR pipeline — layout-preserving JSON/searchable PDF, transliteration, context correction

---

## 2. Knowledge base

| Concept | Plain-language explanation |
|---|---|
| OCR | Optical Character Recognition — turning pixels into machine-readable text |
| LSTM & Tesseract | Tesseract 4/5 reads a line via an LSTM (recurrent neural net using surrounding-character context). Google's open `tessdata_best` weights, loaded locally — no API, no training here |
| CRAFT + CRNN & EasyOCR | Two chained models: CRAFT locates text, a CRNN (CNN + RNN) reads each region. PyTorch, GPU-enabled in this repo |
| PP-OCRv5 & PaddleOCR | Baidu/PaddlePaddle's OCR family — ships purpose-trained recognition models per script family, including an official Devanagari and Tamil model |
| TrOCR | ViT-encoder + transformer-decoder architecture, the handwriting specialist of the four — but see the risk register, its reliability here is unproven/negative |
| CER | Character Error Rate — substitutions + insertions + deletions ÷ reference length, via `jiwer`. The only honest way to compare engines |
| Bounding-box cropping | Each annotated field is cropped (+8px pad) before OCR, so every engine reads one homogeneous strip, not a whole noisy page |
| ULCA schema | Bhashini's data standard: image + per-field bbox + ground truth + language metadata |
| Engine plugin pattern | Every backend implements `is_available_for(script)` / `recognize(crop, script)` — adding one engine never touches the dataset reader or routers |

**Request pipeline** (`POST /api/ocr/run-on-sample`):

```
Annotated field (image + bbox + ground truth)
        │
        ▼
  bbox_crop()  →  isolates the field, +8px pad
        │
        ▼
  engines_for_script()  →  fan out to selected engines
        │             (Tesseract | EasyOCR | PaddleOCR | TrOCR)
        ▼
  safe_cer()  →  each hypothesis scored against ground truth
        │
        ▼
  JSON response → rendered in the UI
```

---

## 3. Project lifecycle

Numbering follows the repo's own convention (this demo already calls itself
"Stage 3," against a "Stage 2" dataset).

| Stage | Name | Status | Notes |
|---|---|---|---|
| 2 | Dataset annotation | ✅ Done | ULCA-schema ground truth in sibling `Akshar_dristhii_code` repo — bboxes, transcriptions, language metadata, 10 scripts |
| 3 | Baseline benchmarking | 🔵 Current | This repo — Tesseract, EasyOCR, PaddleOCR, TrOCR, CER-scored, manual engine selection, GPU-accelerated where possible |
| 4 | Layout, handwriting & model adaptation | ⬜ Planned | No page-level layout detection yet; handwriting accuracy still not production-grade — path forward is fine-tuning PaddleOCR/EasyOCR on the newly-found real dataset, not another off-the-shelf checkpoint |
| 5 | Post-OCR pipeline | ⬜ Planned | Layout-preserving JSON/searchable PDF, transliteration, language/code-mix detection, context-aware correction |
| 6 | Bhashini deployment integration | ⬜ Planned | Stable API contract, containerized GPU inference serving, auth/rate-limiting/observability |

---

## 4. Tech stack

### In place today
| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI · Python 3.12 | Thin REST layer over the dataset reader and OCR engines |
| Frontend | Vanilla JS + static HTML/CSS | No framework overhead for a single-page browser |
| Print OCR | `pytesseract` (Tesseract LSTM) | Fast, broad language coverage via `tessdata_best` |
| Deep-learning OCR | `easyocr` (CRAFT + CRNN) | GPU-enabled |
| Deep-learning OCR | `paddleocr` (PP-OCRv5) | Official Devanagari + Tamil models; CPU here (GPU DLL issue) |
| Handwriting OCR | `transformers` TrOCR (community checkpoint) | Benchmarked and found unreliable — see risks |
| Evaluation | `jiwer` (CER) | Standard, defensible OCR accuracy metric |
| Imaging | `Pillow`, `numpy` | Cropping, array conversion between engines |
| Acceleration | CUDA `torch` | RTX 3050 (4GB) driving EasyOCR + TrOCR |
| Frontend UX | Engine-selection checkboxes | Run any subset of the 4 engines per request |
| Data (demo) | ULCA-schema JSON + images, curated subset | Read-only, 15 real + 90 synthetic fields/script |
| Data (untapped) | Sibling repo's `raw-data/` + `synthetic/` | 10,943 real + 414,000 synthetic labeled fields, unused |

### Needed for full BHASHINI scope
| Layer | Candidate | Closes which gap |
|---|---|---|
| Handwriting | Fine-tune PaddleOCR (decided) on the real dataset | TrOCR-checkpoint shortcut ruled unreliable; PaddleOCR won the head-to-head |
| Document understanding | LayoutLM / Donut / DocTR | Multi-column, tables, fixed-field forms |
| Preprocessing | OpenCV deskew/denoise/contrast | Blur, fold, stamp, skew handling |
| Language ID | Per-region script/language detector | Code-mixed, multi-script documents |
| Post-OCR output | Structured JSON, PDF gen, transliteration lib | Layout-preserving, searchable deliverables |
| MLOps | Experiment tracking (MLflow/W&B) | Once fine-tuning actually starts |
| Inference serving | TorchServe/Triton, containerized | Production-grade GPU serving at scale |
| Orchestration | Prefect/Airflow | Multi-stage document pipelines |
| Ops | Auth, rate limiting, logging/metrics | What a citizen-facing service needs |

---

## 5. Risk & gap register

Same convention the repo's own README follows: surface weaknesses, don't bury them.

- **TrOCR is integrated but unreliable.** No Microsoft-official Indic checkpoint exists; the community checkpoint in use tested worst-of-four on two independent unseen-data tests, and its one strong result is likely training-data overlap, not genuine skill. Kept in the UI as a benchmarked option, not a recommended one.
- **The demo's dataset browser shows a tiny curated slice, not the real data volume.** 10,943 real + 414,000 synthetic labeled fields sit unused in the sibling repo — easy to wrongly conclude "not enough data" without checking there first (we did, initially).
- **Ol Chiki has zero engine support** across all four engines.
- **Gujarati, Malayalam, Odia, Punjabi** are Tesseract-only — no deep-learning engine reaches them today.
- **Real handwriting accuracy is still not good enough**, even from the best current engine — fine-tuning is a plan, not a result yet.
- **No layout or table detection.** Every OCR call operates on one pre-cropped field.
- **GPU acceleration isn't guaranteed portable.** `paddlepaddle-gpu`'s Windows wheel availability is CUDA-tag-dependent.
- **No production hardening.** No auth, rate limiting, or structured logging — fine for a local demo, not for a Bhashini-facing service.
