"""ULCA OCR record building shared by the public /v1 API.

Field names, enums and the per-script language/script codes are copied from
the Stage 2 dataset repo's schemas/ocr-record.schema.json (itself copied
verbatim from github.com/bhashini-dibd/ulca specs) and from that repo's own
already-shipped records -- not invented here. The schema is
additionalProperties: false, so nothing outside it (confidence, engine name,
...) may go inside a record; callers return those alongside the records.

Mirrors the record shape ocr_router.py's create-ground-truth endpoint already
writes, extended to every script the app has an engine for.
"""
from PIL import Image

# Same codes as the dataset repo's synthetic/<script>/kv/clean/*.json records.
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

# PIL format name -> ULCA ImageFormat enum.
IMAGE_FORMAT_MAP = {"JPEG": "jpeg", "PNG": "png", "BMP": "bmp", "TIFF": "tiff"}

IMAGE_TEXT_TYPES = ("computer-typed-text", "handwritten-text", "scene-text", "typewriter-typed-text")

SUBMITTER = {
    "name": "AksharDrishti citizen upload",
    "aboutMe": "BHASHINI AksharDrishti Hackathon Stage 3 demo",
}


def image_format(im: Image.Image) -> str:
    return IMAGE_FORMAT_MAP.get(im.format, "png")


def image_dpi(im: Image.Image) -> str:
    """ULCA ImageDPI only has two buckets; use the file's own DPI tag when it
    has one, else 72_dpi (same default create-ground-truth uses)."""
    dpi = im.info.get("dpi")
    try:
        return "300_dpi" if dpi and float(dpi[0]) >= 300 else "72_dpi"
    except (TypeError, ValueError, IndexError):
        return "72_dpi"


def quad_vertices(bbox) -> list[dict]:
    """[[x, y], ...] polygon -> ULCA Vertex list."""
    return [{"x": round(float(pt[0]), 2), "y": round(float(pt[1]), 2)} for pt in bbox]


def full_image_vertices(im: Image.Image) -> list[dict]:
    """Clockwise corners of the whole image -- the bounding box of a word-level
    (already-cropped) recognition."""
    w, h = im.size
    return quad_vertices([[0, 0], [w, 0], [w, h], [0, h]])


def build_record(script: str, text: str, vertices: list[dict], *, fmt: str, dpi: str,
                 image_filename: str, image_text_type: str, collection_source: str) -> dict:
    return {
        "datasetType": "ocr-corpus",
        "languages": dict(ULCA_LANGUAGE[script]),
        "collectionSource": [collection_source],
        "domain": ["general"],
        "license": "unknown-license",
        "submitter": dict(SUBMITTER),
        "format": fmt,
        "dpi": dpi,
        "imageTextType": image_text_type,
        "imageFilename": image_filename,
        "groundTruth": text,
        "boundingBox": {"vertices": vertices},
    }
