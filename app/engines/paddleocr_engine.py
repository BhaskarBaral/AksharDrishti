import numpy as np
from PIL import Image

from app.engines.base import OCREngine
from app.layout import RTL_SCRIPTS, sort_reading_order

# script folder name -> PaddleOCR --lang code.
# PP-OCRv5 only ships an official (non-community) recognition model for these
# two Indic scripts today -- see docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.
# Devanagari's dedicated model is invoked via lang="hi" (covers Hindi/Marathi/
# Nepali/Sanskrit etc., all Devanagari-script languages share one model).
SCRIPT_TO_LANG = {
    "devanagari": "hi",
    "tamil": "ta",
}


class PaddleOCREngine(OCREngine):
    """Pipelines are loaded lazily and cached per language -- each one is a
    real model load, so we don't want that at import time or per-request."""

    name = "paddleocr"
    supports_page_level = True

    def __init__(self):
        self._pipelines: dict[str, object] = {}
        self._gpu_failed = False

    def is_available_for(self, script: str) -> bool:
        return script in SCRIPT_TO_LANG

    def _get_pipeline(self, lang: str):
        if lang not in self._pipelines:
            from paddleocr import PaddleOCR

            device = "cpu" if self._gpu_failed else "gpu:0"
            # enable_mkldnn=False works around a PaddlePaddle 3.3.x regression where
            # CPU (oneDNN) inference raises NotImplementedError: ConvertPirAttribute2
            # RuntimeAttribute -- https://github.com/PaddlePaddle/PaddleOCR/issues/18162
            try:
                self._pipelines[lang] = PaddleOCR(
                    lang=lang, device=device, enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception:
                # driver/CUDA build mismatch -- fall back to CPU rather than crash
                self._gpu_failed = True
                self._pipelines[lang] = PaddleOCR(
                    lang=lang, device="cpu", enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
        return self._pipelines[lang]

    def recognize(self, crop: Image.Image, script: str) -> str:
        lang = SCRIPT_TO_LANG[script]
        try:
            pipeline = self._get_pipeline(lang)
            # PaddleOCR follows the OpenCV/BGR convention internally.
            bgr = np.array(crop.convert("RGB"))[:, :, ::-1]
            results = pipeline.predict(bgr)
            texts = [t for res in results for t in res["rec_texts"]]
            return " ".join(texts).strip()
        except Exception as e:
            return f"__ERROR__:{e}"

    def recognize_page(self, image: Image.Image, script: str) -> list[dict]:
        lang = SCRIPT_TO_LANG[script]
        pipeline = self._get_pipeline(lang)
        bgr = np.array(image.convert("RGB"))[:, :, ::-1]
        results = pipeline.predict(bgr)

        items = []
        for res in results:
            for poly, text, score in zip(res["rec_polys"], res["rec_texts"], res["rec_scores"]):
                items.append({
                    "bbox": [[float(x), float(y)] for x, y in poly],
                    "text": text,
                    "confidence": float(score),
                })
        return sort_reading_order(items, rtl=script in RTL_SCRIPTS)
