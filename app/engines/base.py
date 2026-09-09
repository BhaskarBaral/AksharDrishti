"""Common interface every OCR backend implements, so the router never
needs to know which engine it's calling."""
from abc import ABC, abstractmethod

from PIL import Image


class OCREngine(ABC):
    name: str

    @abstractmethod
    def is_available_for(self, script: str) -> bool:
        """Whether this engine has a language mapping for the given script folder name."""

    @abstractmethod
    def recognize(self, crop: Image.Image, script: str) -> str:
        """Run OCR on an already-cropped field image, return recognized text."""
