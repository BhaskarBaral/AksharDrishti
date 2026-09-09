from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.routers import dataset_router, ocr_router

app = FastAPI(title="AksharDrishti Stage 3 Demo")

app.include_router(dataset_router.router)
app.include_router(ocr_router.router)

# Serves static/index.html at "/" and static/app.js, static/styles.css alongside it.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
