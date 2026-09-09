import re

import numpy as np
from PIL import Image

from app.config import DEMO_ROOT
from app.engines.base import OCREngine

# PaddleX's modern TextRecognition inference path never reverses RTL output
# on its own (see ppocr/postprocess/rec_postprocess.py's legacy "arabic" in
# character_dict_path check -- that heuristic only fires during the older
# training/eval tooling, not this predictor). The official Arabic checkpoint
# looks correct anyway because its own training labels were apparently
# already stored in CTC-native (visual, left-to-right-scan) order. Ours
# fine-tuned on labels in normal logical order instead -- CTC decoding is
# strictly monotonic (left-to-right by construction, can't reorder), so the
# network converged on emitting reversed-order text. Confirmed empirically
# (both held-out and training-set predictions match ground truth almost
# exactly once reversed), so undo it here with PaddleOCR's own pred_reverse
# algorithm, which protects embedded Latin/digit runs from being flipped.
_RTL_SCRIPTS = {"urdu"}


def _pred_reverse(pred: str) -> str:
    pred_re = []
    c_current = ""
    for c in pred:
        if not re.search("[a-zA-Z0-9 :*./%+-]", c):
            if c_current:
                pred_re.append(c_current)
            pred_re.append(c)
            c_current = ""
        else:
            c_current += c
    if c_current:
        pred_re.append(c_current)
    return "".join(pred_re[::-1])

TRAINING_OUTPUT_DIR = DEMO_ROOT / "training" / "output"

# script -> PaddleX registered model name the checkpoint was fine-tuned from.
# Must match training/generate_configs.py's BASE_TEMPLATE (minus ".yaml") --
# the scripts with no official PP-OCRv5 checkpoint were all fine-tuned from
# the Devanagari architecture as a generic Indic-abugida backbone.
SCRIPT_TO_MODEL_NAME = {
    "devanagari": "devanagari_PP-OCRv5_mobile_rec",
    "tamil": "ta_PP-OCRv5_mobile_rec",
    "telugu": "te_PP-OCRv5_mobile_rec",
    "urdu": "arabic_PP-OCRv5_mobile_rec",
    "bengali": "devanagari_PP-OCRv5_mobile_rec",
    "gujarati": "devanagari_PP-OCRv5_mobile_rec",
    "kannada": "devanagari_PP-OCRv5_mobile_rec",
    "malayalam": "devanagari_PP-OCRv5_mobile_rec",
    "odia": "devanagari_PP-OCRv5_mobile_rec",
    "punjabi": "devanagari_PP-OCRv5_mobile_rec",
}


class PaddleOCRFinetunedEngine(OCREngine):
    """Our own fine-tuned checkpoints (training/output/<script>/best_accuracy),
    as a separate engine from the official-model PaddleOCREngine so the UI can
    show both side by side. CPU only -- the app's .venv intentionally carries
    CPU-only paddlepaddle (see training/run_all.py's module docstring for why
    a GPU build can't safely share this process with torch)."""

    name = "paddleocr-finetuned"

    def __init__(self):
        self._predictors: dict[str, object] = {}

    def is_available_for(self, script: str) -> bool:
        return script in SCRIPT_TO_MODEL_NAME and self._model_dir(script).exists()

    def _model_dir(self, script: str):
        return TRAINING_OUTPUT_DIR / script / "best_accuracy" / "inference"

    def _get_predictor(self, script: str):
        if script not in self._predictors:
            from paddleocr import TextRecognition

            self._predictors[script] = TextRecognition(
                model_name=SCRIPT_TO_MODEL_NAME[script],
                model_dir=str(self._model_dir(script)),
                device="cpu",
            )
        return self._predictors[script]

    def recognize(self, crop: Image.Image, script: str) -> str:
        try:
            predictor = self._get_predictor(script)
            bgr = np.array(crop.convert("RGB"))[:, :, ::-1]
            result = next(iter(predictor.predict(bgr)))
            text = result["rec_text"]
            return _pred_reverse(text) if script in _RTL_SCRIPTS else text
        except Exception as e:
            return f"__ERROR__:{e}"
