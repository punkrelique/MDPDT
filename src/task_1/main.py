import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image

import services.models.caption as captioner
import services.models.llm as llm
import services.models.moderate as moderate
import services.models.tts as tts
import services.models.video as video_detector
from services.pipelines import SensitiveContentError, ask, describe_video
from services.utils import is_russian


@asynccontextmanager
async def lifespan(_app: FastAPI):
    moderate.load()
    captioner.load()
    llm.load()
    video_detector.load()
    tts.load()
    yield
    tts.unload()
    video_detector.unload()
    llm.unload()
    captioner.unload()
    moderate.unload()


app = FastAPI(title="Image QA API", lifespan=lifespan)


@app.get("/")
def root():
    return RedirectResponse("/docs")


@app.post("/v1/image/ask")
def image_ask(question: str = Form(..., min_length=1), image: UploadFile = File(...)):
    if not is_russian(question):
        raise HTTPException(status_code=422, detail="Only Russian questions are allowed")
    try:
        picture = Image.open(io.BytesIO(image.file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc
    try:
        return ask(picture, question)
    except SensitiveContentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/video/detect")
def video_detect(video: UploadFile = File(...)):
    try:
        payload = video.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid video") from exc
    try:
        return video_detector.detect_objects(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/video/describe")
def video_describe(video: UploadFile = File(...), voice: str = Form("default")):
    try:
        payload = video.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid video") from exc
    try:
        return describe_video(payload, voice=voice)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
