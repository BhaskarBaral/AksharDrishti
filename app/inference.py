"""State shared by every router that runs models (OCR tab, /v1 API, Layout)."""
import threading

# One model inference at a time across the whole app. Sync endpoints run in
# FastAPI's threadpool, and two overlapping requests put paddle and torch
# inference on separate threads at once -- that segfaulted the whole server
# (exit 139) in testing 2026-10-27, which a single-user demo can't afford.
# Requests queue instead; per-request latency is unchanged. Must be one lock
# app-wide: a per-router lock still lets two tabs overlap.
INFERENCE_LOCK = threading.Lock()
