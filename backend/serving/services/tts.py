"""Text-to-speech: sentence -> WAV bytes in-memory (no disk writes)."""

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from serving.config import settings
from serving.utils.logger import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)


class TTSService:
    def __init__(self):
        self._engine = None

    def load(self) -> None:
        from vieneu import Vieneu

        logger.info("Dang nap VieNeu-TTS (lan dau co the lau)...")
        self._engine = Vieneu(mode=settings.tts_mode)

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def sample_rate(self) -> int:
        return getattr(self._engine, "sample_rate", settings.tts_sample_rate)

    def _synthesize_sync(self, text: str) -> bytes:
        audio = np.asarray(self._engine.infer(text), dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("TTS tra ve audio rong")
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()

    async def synthesize(self, text: str) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._synthesize_sync, text)
