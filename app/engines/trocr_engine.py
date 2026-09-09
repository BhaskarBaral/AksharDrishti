from PIL import Image

from app.engines.base import OCREngine

# script folder name -> Hugging Face repo id.
# The only handwriting-capable engine of the four. No official Microsoft TrOCR
# checkpoint exists for any Indic script, so this is a community fine-tune
# (trained on handwritten Devanagari/Nepali data) rather than a vetted release --
# the model card states no license, which is flagged here rather than hidden.
SCRIPT_TO_MODEL = {
    "devanagari": "aayushpuri01/TrOCR-Devanagari",
}


class TrOCREngine(OCREngine):
    """Models are loaded lazily and cached per script -- each is a real model
    load, so it must not happen at import time or per-request."""

    name = "trocr"

    def __init__(self):
        self._models: dict[str, tuple] = {}

    def is_available_for(self, script: str) -> bool:
        return script in SCRIPT_TO_MODEL

    def _get_model(self, script: str):
        if script not in self._models:
            import torch
            from transformers import RobertaTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel

            repo = SCRIPT_TO_MODEL[script]
            # This checkpoint ships legacy vocab.json+merges.txt (no tokenizer.json),
            # and current transformers' AutoTokenizer routes everything through its
            # Rust "fast" backend, which fails to auto-convert this format ("Couldn't
            # instantiate the backend tokenizer... need sentencepiece or tiktoken" --
            # a red herring; sentencepiece is installed). Loading the explicit
            # RobertaTokenizer class (matching this repo's GPT2/RoBERTa-style BPE
            # format) bypasses that broken auto-conversion path.
            # Similarly, this checkpoint's own preprocessor_config.json isn't
            # recognized ("Unrecognized image processor"), so the image processor is
            # built from the base ViT encoder it was fine-tuned from instead of
            # trusting the repo's own (stale) config.
            tokenizer = RobertaTokenizer.from_pretrained(repo)
            image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
            model = VisionEncoderDecoderModel.from_pretrained(repo)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            self._models[script] = (processor, model, device)
        return self._models[script]

    def recognize(self, crop: Image.Image, script: str) -> str:
        try:
            import torch

            processor, model, device = self._get_model(script)
            pixel_values = processor(crop.convert("RGB"), return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=64)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            return f"__ERROR__:{e}"
