"""Single place that knows which engine instances exist. Adding a new
backend later means writing one OCREngine subclass and adding it here --
routers and the dataset layer are untouched."""
from functools import lru_cache

from app.engines.base import OCREngine
from app.engines.bhashini_ocr_engine import BhashiniOCREngine
from app.engines.easyocr_engine import EasyOCREngine
from app.engines.paddleocr_engine import PaddleOCREngine
from app.engines.paddleocr_finetuned_engine import PaddleOCRFinetunedEngine
from app.engines.tesseract_engine import TesseractEngine
from app.engines.trocr_engine import TrOCREngine


@lru_cache(maxsize=1)
def get_engines() -> list[OCREngine]:
    return [
        TesseractEngine(),
        EasyOCREngine(),
        PaddleOCREngine(),
        PaddleOCRFinetunedEngine(),
        TrOCREngine(),
        BhashiniOCREngine(),
    ]


def engines_for_script(script: str) -> list[OCREngine]:
    return [e for e in get_engines() if e.is_available_for(script)]
