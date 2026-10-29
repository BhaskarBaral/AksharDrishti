from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.routers import api_v1_router, dataset_router, layout_router, ocr_router

app = FastAPI(title="AksharDrishti Stage 3 Demo")

app.include_router(dataset_router.router)
app.include_router(ocr_router.router)
app.include_router(layout_router.router)
app.include_router(api_v1_router.router)

# Serves static/index.html at "/" and static/app.js, static/styles.css alongside it.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
