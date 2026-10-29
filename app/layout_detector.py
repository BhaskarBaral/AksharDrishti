"""Page-layout detector: two DocLayout-YOLO models combined.

- Structure (IndicDLP's 42 classes: titles, paragraphs, lists, tables, form
  placeholders, figures, headers/footers...) comes from the IndicDLP authors'
  own checkpoint, trained on ~120k real Indic pages (CC-BY-4.0).
- seal / stamp / signature, which IndicDLP doesn't have, come from our
  fine-tune of that checkpoint (training/layout/train_yolo.py: real IndicDLP
  pages + Stage 2 synthetic pages), and only those three classes.

Why not the fine-tune alone: on IndicDLP's real held-out Odia test pages it
scored mAP@0.5 0.446 vs the original's 0.746 (training/layout/runs/
eval_finetuned.json, evaluate_yolo.log) -- 3.4k real pages at 800 px wasn't
enough to keep the original's real-page skill -- while it does learn the new
classes (synthetic test AP@0.5: seal 0.994, stamp 0.993, signature 0.938).

Separate from layout.py, which only orders OCR lines -- this finds the regions."""
import os
from pathlib import Path

from PIL import Image

from app.config import DEMO_ROOT

# Weights live in weights/ (tracked with Git LFS -- run `git lfs pull` after cloning).
# Override with INDICDLP_WEIGHTS / LAYOUT_FT_WEIGHTS to point elsewhere.
BASE = Path(os.environ.get("INDICDLP_WEIGHTS",
                           DEMO_ROOT / "weights" / "indicdlp" / "doclayout_yolo_indicdlp.pt"))
FINETUNED = Path(os.environ.get("LAYOUT_FT_WEIGHTS",
                                DEMO_ROOT / "weights" / "layout_ft" / "best.pt"))
BASE_IMGSZ = 1024  # the IndicDLP authors' training size
FINETUNED_IMGSZ = 800  # train_yolo.py's size (1024 overflows a 4 GB GPU)
EXTRA_CLASSES = {"seal", "stamp", "signature"}
# The extras' only training examples are synthetic; at 0.5 the fine-tune
# fired 19 "stamps" on the 325 real complex-testset forms, most of them
# boxed text. Keep them stricter than the structure classes.
EXTRA_MIN_SCORE = 0.7


class LayoutDetector:
    def __init__(self):
        self._base = None
        self._extras = None

    @property
    def model_name(self) -> str:
        if FINETUNED.exists():
            return "IndicDLP DocLayout-YOLO (real pages) + our fine-tune for seals/stamps/signatures"
        return "IndicDLP DocLayout-YOLO (real pages)"

    def is_available(self) -> bool:
        return BASE.exists()

    def _models(self):
        from doclayout_yolo import YOLOv10

        if self._base is None:
            self._base = YOLOv10(str(BASE))
        if self._extras is None and FINETUNED.exists():
            self._extras = YOLOv10(str(FINETUNED))
        return self._base, self._extras

    def detect(self, image: Image.Image, score_threshold: float = 0.3) -> list[dict]:
        """-> [{"class", "bbox": [x0, y0, x1, y1] in original pixels, "score"}],
        sorted top-to-bottom."""
        import numpy as np
        import torch

        base, extras = self._models()
        bgr = np.array(image.convert("RGB"))[:, :, ::-1]
        device = 0 if torch.cuda.is_available() else "cpu"
        regions = []
        res = base.predict(bgr, imgsz=BASE_IMGSZ, conf=score_threshold, device=device, verbose=False)[0]
        regions += _regions(res, keep=lambda cls: cls not in EXTRA_CLASSES)
        if extras is not None:
            min_score = max(score_threshold, EXTRA_MIN_SCORE)
            res = extras.predict(bgr, imgsz=FINETUNED_IMGSZ, conf=min_score, device=device, verbose=False)[0]
            regions += _regions(res, keep=lambda cls: cls in EXTRA_CLASSES)
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return regions


def _regions(res, keep) -> list[dict]:
    out = []
    for box, c, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        cls = res.names[int(c)]
        if keep(cls):
            out.append({"class": cls, "bbox": [round(v, 1) for v in box], "score": round(score, 4)})
    return out


detector = LayoutDetector()
