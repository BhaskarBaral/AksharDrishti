import pytesseract
from PIL import Image

from app.config import TESSDATA_DIR, TESSERACT_EXE
from app.engines.base import OCREngine
from app.layout import RTL_SCRIPTS, sort_reading_order

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
    # Tesseract does its own layout analysis (--psm 3), so it can report
    # per-line boxes too -- the only page-capable engine with a language pack
    # for every one of the 10 scripts, and the most accurate one on printed
    # pages for the 8 scripts stock PaddleOCR doesn't cover.
    supports_page_level = True

    def is_available_for(self, script: str) -> bool:
        return script in SCRIPT_TO_LANG

    def recognize(self, crop: Image.Image, script: str, is_full_page: bool = False) -> str:
        lang = SCRIPT_TO_LANG[script]
        # --psm 7 ("treat the image as a single text line") is right for a
        # tight field crop (run-on-sample), but silently returns empty on a
        # whole page: Tesseract's layout analysis doesn't find one line
        # spanning the full image, so it finds nothing at all rather than
        # falling back to reading what's there. A raw upload has no crop to
        # give it, so it needs Tesseract's own general page segmentation
        # (--psm 3) instead -- this was the actual cause of "(empty)" results
        # on whole-page uploads, not a missing language pack or bad image.
        psm = 3 if is_full_page else 7
        try:
            return pytesseract.image_to_string(
                crop, lang=lang,
                config=f"--tessdata-dir {TESSDATA_DIR.as_posix()} --psm {psm}",
            ).strip()
        except Exception as e:
            return f"__ERROR__:{e}"

    def recognize_page(self, image: Image.Image, script: str) -> list[dict]:
        lang = SCRIPT_TO_LANG[script]
        data = pytesseract.image_to_data(
            image.convert("RGB"), lang=lang,
            config=f"--tessdata-dir {TESSDATA_DIR.as_posix()} --psm 3",
            output_type=pytesseract.Output.DICT,
        )
        # image_to_data is word-level; group words back into Tesseract's own
        # (block, paragraph, line) lines, bbox = union of the word boxes,
        # confidence = mean word confidence (0-100 -> 0-1).
        lines: dict[tuple, dict] = {}
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            line = lines.setdefault(key, {"words": [], "confs": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h})
            line["words"].append(word)
            line["confs"].append(conf)
            line["x0"], line["y0"] = min(line["x0"], x), min(line["y0"], y)
            line["x1"], line["y1"] = max(line["x1"], x + w), max(line["y1"], y + h)

        items = [
            {
                "bbox": [[l["x0"], l["y0"]], [l["x1"], l["y0"]], [l["x1"], l["y1"]], [l["x0"], l["y1"]]],
                "text": " ".join(l["words"]),
                "confidence": sum(l["confs"]) / len(l["confs"]) / 100.0,
            }
            for l in lines.values()
        ]
        return sort_reading_order(items, rtl=script in RTL_SCRIPTS)
