from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.dataset import reader

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.get("/scripts")
def scripts():
    return {"scripts": reader.list_scripts()}


@router.get("/scripts/{script}/samples")
def samples(script: str):
    result = reader.list_samples(script)
    if not result:
        raise HTTPException(404, f"No annotated samples for script {script!r}")
    return {"script": script, "samples": result}


@router.get("/scripts/{script}/{split}/{filename}/records")
def records(script: str, split: str, filename: str):
    try:
        return {"records": reader.get_records(script, split, filename)}
    except reader.SampleNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/scripts/{script}/{split}/{filename}/image")
def image(script: str, split: str, filename: str):
    try:
        path = reader.resolve_image_path(script, split, filename)
    except reader.SampleNotFoundError as e:
        raise HTTPException(404, str(e))
    return FileResponse(path)
