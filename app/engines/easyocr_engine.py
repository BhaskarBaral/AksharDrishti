import numpy as np
from PIL import Image

from app.engines.base import OCREngine

# script folder name -> easyocr language code
SCRIPT_TO_LANG = {
    "devanagari": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "urdu": "ur",
    "kannada": "kn",
    "telugu": "te",
    # gujarati/malayalam/odia/punjabi/olchiki: no easyocr language support
}


class EasyOCREngine(OCREngine):
    """Readers are loaded lazily and cached per language -- each one is a
    real model load, so we don't want that at import time or per-request."""

    name = "easyocr"

    def __init__(self):
        self._readers: dict[str, object] = {}

    def is_available_for(self, script: str) -> bool:
        return script in SCRIPT_TO_LANG

    def _get_reader(self, lang: str):
        if lang not in self._readers:
            import easyocr
            self._readers[lang] = easyocr.Reader([lang, "en"], gpu=True, verbose=False)
        return self._readers[lang]

    def recognize(self, crop: Image.Image, script: str) -> str:
        lang = SCRIPT_TO_LANG[script]
        try:
            reader = self._get_reader(lang)
            result = reader.readtext(np.array(crop.convert("RGB")), detail=0)
            return " ".join(result).strip()
        except Exception as e:
            return f"__ERROR__:{e}"
