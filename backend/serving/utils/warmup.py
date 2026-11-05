"""Startup warmup: load SPOTER + TTS, and warm up Ollama."""

import asyncio

import numpy as np

from serving.config import settings
from serving.dependencies import (
    get_gloss_decoder,
    get_llm,
    get_slgcn,
    get_spoter,
    get_tts,
    get_video_recognizer,
)
from serving.utils.logger import get_logger

logger = get_logger(__name__)


async def ensure_tts_loaded() -> None:
    """Loads TTS once (idempotent); used at warmup and by handlers on demand."""
    tts = get_tts()
    if not tts.is_loaded:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, tts.load)


async def warmup_all() -> None:
    loop = asyncio.get_running_loop()

    # load() is CPU/GPU-bound; run in an executor so it doesn't block the event loop.
    await loop.run_in_executor(None, get_gloss_decoder().load)
    await loop.run_in_executor(None, get_spoter().load)
    # SL-GCN (2nd fusion branch): video flow only, but loaded early to warm it up.
    try:
        await loop.run_in_executor(None, get_slgcn().load)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nap SL-GCN that bai: %s", exc)

    # VideoRecognizer reuses SpoterService/SlgcnService's sessions -> load after both.
    try:
        await loop.run_in_executor(None, get_video_recognizer().load)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nap VideoRecognizer that bai: %s", exc)

    # Run SPOTER on dummy input to warm up the graph/CUDA context.
    dummy = np.zeros((settings.spoter_seq_len, 54, 2), dtype=np.float32)
    try:
        await get_spoter().infer(dummy)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warmup SPOTER that bai: %s", exc)

    # Run SL-GCN on dummy input (1, 3, slgcn_seq_len, num_points, 1).
    slgcn = get_slgcn()
    if slgcn.session is not None:
        slgcn_dummy = np.zeros(
            (1, 3, settings.slgcn_seq_len, settings.slgcn_num_points, 1), dtype=np.float32
        )
        try:
            await loop.run_in_executor(None, slgcn.probs, slgcn_dummy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warmup SL-GCN that bai: %s", exc)

    get_llm().load()
    try:
        await get_llm().ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama chua san sang (%r). Cau se ghep tu tho khi thieu LLM.", exc)

    # TTS loads a large HuggingFace model on first use; guard against startup crashes.
    try:
        await ensure_tts_loaded()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nap TTS that bai (%s). Se thu lai khi co request TTS.", exc)

    logger.info("Warmup hoan tat.")
