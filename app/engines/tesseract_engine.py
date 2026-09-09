import pytesseract
from PIL import Image

from app.config import TESSDATA_DIR, TESSERACT_EXE
from app.engines.base import OCREngine

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# script folder name -> tesseract language code (tessdata_best packs in TESSDATA_DIR)
SCRIPT_TO_LANG = {
    "devanagari": "hin",
    "bengali": "ben",
    "tamil": "tam",
    "urdu": "urd",
    "gujarati": "guj",
    "kannada": "kan",
    "malayalam": "mal",
    "odia": "ori",
    "punjabi": "pan",
    "telugu": "tel",
    # olchiki: no tesseract language pack exists -- documented gap, not a bug
}


class TesseractEngine(OCREngine):
    name = "tesseract"

    def is_available_for(self, script: str) -> bool:
        return script in SCRIPT_TO_LANG

    def recognize(self, crop: Image.Image, script: str) -> str:
        lang = SCRIPT_TO_LANG[script]
        try:
            return pytesseract.image_to_string(
                crop, lang=lang,
                config=f"--tessdata-dir {TESSDATA_DIR.as_posix()} --psm 7",
            ).strip()
        except Exception as e:
            return f"__ERROR__:{e}"
