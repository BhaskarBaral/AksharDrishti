"""Common interface every OCR backend implements, so the router never
needs to know which engine it's calling."""
from abc import ABC, abstractmethod

from PIL import Image


class OCREngine(ABC):
    name: str

    # Whether recognize_page() is implemented -- only engines with their own
    # text detector (not pure recognition-only ones) can support this.
    supports_page_level: bool = False

    @abstractmethod
    def is_available_for(self, script: str) -> bool:
        """Whether this engine has a language mapping for the given script folder name."""

    @abstractmethod
    def recognize(self, crop: Image.Image, script: str, is_full_page: bool = False) -> str:
        """Run OCR on an already-cropped field image, return recognized text.
        `is_full_page=True` means the caller couldn't crop first (e.g. a raw
        citizen upload) and this is a whole page, not a tight single-field
        crop -- engines whose recognition-only path assumes a single line
        (Tesseract) need to know this to pick a sane page-segmentation mode.
        Engines that always do their own detection regardless of input size
        (EasyOCR, PaddleOCR) can ignore the flag."""

    def recognize_page(self, image: Image.Image, script: str) -> list[dict]:
        """Detect and recognize every text line in a full (uncropped) image,
        returned in reading order as {"bbox", "text", "confidence"} dicts.
        Only implemented by engines with supports_page_level = True."""
        raise NotImplementedError(f"{self.name} does not support page-level extraction")
