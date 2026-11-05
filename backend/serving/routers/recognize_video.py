"""Video-file recognition + sentence composition. POST /api/recognize-video: video ->
word list (no LLM/TTS). POST /api/compose: word list -> sentence (LLM) + voice (TTS)."""

import base64
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from serving.config import settings
from serving.dependencies import get_llm, get_tts, get_video_recognizer
from serving.models.schemas import ComposeRequest, ComposeResponse, VideoRecognizeResponse
from serving.utils.logger import get_logger
from serving.utils.warmup import ensure_tts_loaded

router = APIRouter()
logger = get_logger(__name__)


@router.post("/recognize-video", response_model=VideoRecognizeResponse)
async def recognize_video(file: UploadFile = File(...)):
    """One video file -> recognized word list (auto-segments multiple signs)."""
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=415, detail="Chi chap nhan file video (video/*).")

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"File vuot qua {settings.max_upload_mb}MB."
        )

    # cv2/ffmpeg pick a demuxer by file extension -> keep the original suffix.
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        glosses = await get_video_recognizer().recognize(tmp.name)
    except ValueError as exc:  # corrupt video / no readable frames
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return VideoRecognizeResponse(glosses=glosses)


@router.post("/compose", response_model=ComposeResponse)
async def compose(req: ComposeRequest):
    """Word list -> Vietnamese sentence (LLM) + voice (TTS); each step fails safely."""
    try:
        sentence = await get_llm().gloss_to_sentence(req.glosses)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM loi (%r); ghep tu tho.", exc)
        sentence = " ".join(req.glosses)
    if not sentence.strip():
        sentence = " ".join(req.glosses)

    audio_b64 = None
    try:
        await ensure_tts_loaded()
        wav_bytes = await get_tts().synthesize(sentence)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS loi (%r); tra ve cau khong kem giong noi.", exc)

    return ComposeResponse(sentence=sentence, audio_b64=audio_b64)
