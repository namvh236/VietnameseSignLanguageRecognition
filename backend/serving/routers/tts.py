"""POST /api/tts -- returns WAV bytes (fallback for standalone TTS clients)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from serving.dependencies import get_tts
from serving.models.schemas import TTSRequest
from serving.utils.warmup import ensure_tts_loaded

router = APIRouter()


@router.post("/tts")
async def synthesize(req: TTSRequest):
    await ensure_tts_loaded()
    try:
        wav_bytes = await get_tts().synthesize(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return Response(content=wav_bytes, media_type="audio/wav")
