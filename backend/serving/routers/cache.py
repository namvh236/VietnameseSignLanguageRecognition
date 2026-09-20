"""Cache ket qua dich theo file video -- luoi an toan cho demo.

- GET  /api/cache/{key}: video nay da tung xu ly chua? -> tra ve tu + cau + WAV.
- POST /api/cache/{key}: luu ket qua vua chay xong de lan sau phat lai ngay.

`key` = sha256 noi dung file (client tinh). Cung mot file = cung key, ke ca doi ten.
"""

import base64

from fastapi import APIRouter, HTTPException

from serving.config import settings
from serving.dependencies import get_result_cache
from serving.models.schemas import CachedResultPayload, CachedResultResponse
from serving.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _check_enabled(key: str) -> None:
    if not settings.result_cache_enabled:
        raise HTTPException(status_code=404, detail="Cache dang tat.")
    if not get_result_cache().is_valid_key(key):
        raise HTTPException(status_code=400, detail="Key khong hop le (chi nhan hex 16-64 ky tu).")


@router.get("/cache/{key}", response_model=CachedResultResponse)
async def get_cached(key: str):
    _check_enabled(key)
    entry = get_result_cache().get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail="Video nay chua tung duoc xu ly.")
    wav_bytes = entry.get("wav_bytes")
    return CachedResultResponse(
        glosses=entry["glosses"],
        sentence=entry["sentence"],
        duration=entry.get("duration"),
        audio_b64=base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else None,
        source=entry.get("source", "live"),
    )


@router.post("/cache/{key}", status_code=204)
async def put_cached(key: str, payload: CachedResultPayload):
    _check_enabled(key)
    wav_bytes = None
    if payload.audio_b64:
        try:
            wav_bytes = base64.b64decode(payload.audio_b64, validate=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"audio_b64 khong hop le: {exc}")
    get_result_cache().put(
        key,
        glosses=[g.model_dump() for g in payload.glosses],
        sentence=payload.sentence,
        duration=payload.duration,
        wav_bytes=wav_bytes,
        source="live",
    )
