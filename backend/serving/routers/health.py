"""GET /health -- compact status: status + provider + loaded models. `status` is
"ok" only when SPOTER and TTS are loaded and Ollama has the model."""

import httpx
from fastapi import APIRouter

from serving.config import settings
from serving.dependencies import get_spoter, get_tts

router = APIRouter()


async def _ollama_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
        resp.raise_for_status()
        names = [m.get("name") for m in resp.json().get("models", [])]
        return settings.ollama_model in names
    except Exception:  # noqa: BLE001
        return False


@router.get("/health")
async def health():
    spoter = get_spoter()
    tts = get_tts()
    providers = spoter.session.get_providers() if spoter.session else []
    ready = spoter.session is not None and tts.is_loaded and await _ollama_ready()
    return {
        "status": "ok" if ready else "degraded",
        "provider": providers[0] if providers else None,
        "spoter": settings.spoter_onnx_name,
        "llm": settings.ollama_model,
        "tts": f"VieNeu-TTS ({settings.tts_mode})",
    }
