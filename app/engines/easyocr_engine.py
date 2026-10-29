import numpy as np
from PIL import Image

from app.engines.base import OCREngine
from app.layout import RTL_SCRIPTS, sort_reading_order

# script folder name -> easyocr language code
SCRIPT_TO_LANG = {
    "devanagari": "hi",
    "bengali": "bn",
    # tamil: dropped -- easyocr 1.7.2's tamil.pth checkpoint has 143 output
    # classes but its bundled Tamil character list builds a 127-class head, so
    # Reader(["ta"]) always fails with a state_dict size mismatch (every
    # request came back as "__ERROR__:..."). Re-add once easyocr ships a
    # matching pair.
    "urdu": "ur",
    "kannada": "kn",
    "telugu": "te",
    # gujarati/malayalam/odia/punjabi/olchiki: no easyocr language support
}


class EasyOCREngine(OCREngine):
    """Readers are loaded lazily and cached per language -- each one is a
    real model load, so we don't want that at import time or per-request."""

    name = "easyocr"
    supports_page_level = True

    def __init__(self):
        self._readers: dict[str, object] = {}

    def is_available_for(self, script: str) -> bool:
        return script in SCRIPT_TO_LANG

    def _get_reader(self, lang: str):
        if lang not in self._readers:
            import easyocr
            self._readers[lang] = easyocr.Reader([lang, "en"], gpu=True, verbose=False)
        return self._readers[lang]

    def recognize(self, crop: Image.Image, script: str, is_full_page: bool = False) -> str:
        # already does full CRAFT-based detection regardless of input size,
        # so a whole page needs no special-casing here.
        lang = SCRIPT_TO_LANG[script]
        try:
            reader = self._get_reader(lang)
            result = reader.readtext(np.array(crop.convert("RGB")), detail=0)
            return " ".join(result).strip()
        except Exception as e:
            return f"__ERROR__:{e}"

    def recognize_page(self, image: Image.Image, script: str) -> list[dict]:
        lang = SCRIPT_TO_LANG[script]
        reader = self._get_reader(lang)
        raw = reader.readtext(np.array(image.convert("RGB")), detail=1)

        items = [
            {
                "bbox": [[float(x), float(y)] for x, y in box],
                "text": text,
                "confidence": float(confidence),
            }
            for box, text, confidence in raw
        ]
        return sort_reading_order(items, rtl=script in RTL_SCRIPTS)
