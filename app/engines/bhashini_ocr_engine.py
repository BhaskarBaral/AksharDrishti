"""Consumer-direction integration: calls Bhashini's hosted OCR model over
the network instead of running inference locally, so it can be benchmarked
against the local engines on the same dataset (see CHARTER.md's Stage 6
note on Bhashini deployment integration).

Needs real ULCA credentials -- BHASHINI_USER_ID / BHASHINI_ULCA_API_KEY /
BHASHINI_PIPELINE_ID, read from a gitignored .env file (see .env.example)
via app/config.py's load_dotenv() call. Get them from
https://bhashini.gov.in/ulca (My Profile -> API Keys) plus a pipeline
you've subscribed to that includes an OCR task. Without them,
is_available_for() reports this engine as absent rather than failing
per-request, so the rest of the app works unmodified.

Confirmed live OCR service IDs in the ULCA catalog (bhashini.gitbook.io/
bhashini-apis/available-models-for-usage) as of 2026-09 -- when picking a
pipeline to subscribe to, look for one of these under its OCR task:
  - bhashini/iiith-bhasha-ocr        printed text, ~22 languages (IIIT-H)
  - bhashini/iiith/ocr-hw-bhaasha    handwritten, 8 languages    (IIIT-H)
  - bhashini/iiith-ocr-sceneText-all scene text, 13 languages    (IIIT-H)
  - bhashini/bodhan/indic-doc/ocr    printed + handwritten       (Bodhan.AI)
iiith-bhasha-ocr is the closest match to this project's printed annotated
fields; a pipeline bundling ocr-hw-bhaasha would be the one to compare
against the handwriting samples instead.

Bhashini's public API docs only publish the request/response schema for
asr/translation/tts in detail -- OCR is a supported taskType but its exact
field names aren't published. The payload below is inferred from the
documented tasks plus the ULCA schema this project's own dataset already
uses (image + bbox + groundTruth + language, see app/dataset/reader.py).
Confirm the real field names against a live getModelsPipeline response
(the pipelineResponseConfig it returns documents that pipeline's actual
input/output shape) before depending on this for anything beyond a
demo/benchmark.
"""
import base64
import io

import requests
from PIL import Image

from app.config import (
    BHASHINI_CONFIG_URL,
    BHASHINI_PIPELINE_ID,
    BHASHINI_ULCA_API_KEY,
    BHASHINI_USER_ID,
)
from app.engines.base import OCREngine

# script folder name -> ISO 639 language code Bhashini's pipelines expect
SCRIPT_TO_LANG = {
    "devanagari": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
    "urdu": "ur",
}


class BhashiniOCREngine(OCREngine):
    name = "bhashini-ocr"

    def __init__(self):
        # lang -> (compute endpoint url, auth header dict, serviceId)
        self._pipeline_cache: dict[str, tuple[str, dict, str]] = {}

    def _configured(self) -> bool:
        return bool(BHASHINI_USER_ID and BHASHINI_ULCA_API_KEY and BHASHINI_PIPELINE_ID)

    def is_available_for(self, script: str) -> bool:
        return self._configured() and script in SCRIPT_TO_LANG

    def _get_pipeline(self, lang: str) -> tuple[str, dict, str]:
        if lang in self._pipeline_cache:
            return self._pipeline_cache[lang]

        resp = requests.post(
            BHASHINI_CONFIG_URL,
            headers={"userID": BHASHINI_USER_ID, "ulcaApiKey": BHASHINI_ULCA_API_KEY},
            json={
                "pipelineTasks": [
                    {"taskType": "ocr", "config": {"language": {"sourceLanguage": lang}}}
                ],
                "pipelineRequestConfig": {"pipelineId": BHASHINI_PIPELINE_ID},
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        endpoint_info = data["pipelineInferenceAPIEndPoint"]
        endpoint = endpoint_info["callbackUrl"]
        auth = endpoint_info["inferenceApiKey"]
        auth_header = {auth["name"]: auth["value"]}
        service_id = data["pipelineResponseConfig"][0]["config"][0]["serviceId"]

        pipeline = (endpoint, auth_header, service_id)
        self._pipeline_cache[lang] = pipeline
        return pipeline

    def recognize(self, crop: Image.Image, script: str) -> str:
        lang = SCRIPT_TO_LANG[script]
        try:
            endpoint, auth_header, service_id = self._get_pipeline(lang)

            buf = io.BytesIO()
            crop.convert("RGB").save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            resp = requests.post(
                endpoint,
                headers={"Content-Type": "application/json", **auth_header},
                json={
                    "pipelineTasks": [
                        {
                            "taskType": "ocr",
                            "config": {
                                "language": {"sourceLanguage": lang},
                                "serviceId": service_id,
                            },
                        }
                    ],
                    "inputData": {"image": [{"imageContent": image_b64}]},
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["pipelineResponse"][0]["output"][0]["source"].strip()
        except Exception as e:
            return f"__ERROR__:{e}"
