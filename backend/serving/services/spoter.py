"""SPOTER (ONNX) inference wrapper. The client already ran MediaPipe Holistic + joint
selection, so only normalization + ONNX inference remain; spoter_v3.onnx bakes in
softmax, so "probabilities" is used as-is. Runs in a ThreadPoolExecutor."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import torch

from serving.config import settings
from serving.recognition_bridge import load_transforms
from serving.utils.logger import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


class SpoterService:
    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self._transforms = None

    def load(self) -> None:
        self._transforms = load_transforms()
        # Helps onnxruntime-gpu find CUDA/cuDNN installed via pip (nvidia-*-cu12).
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                logger.warning("preload_dlls that bai, bo qua.")
        available = ort.get_available_providers()
        providers = [p for p in settings.spoter_providers if p in available]
        self.session = ort.InferenceSession(str(settings.spoter_onnx_path), providers=providers)
        logger.info("SPOTER san sang. ONNX providers: %s", self.session.get_providers())

    def _probs_sync(self, snapshot: np.ndarray) -> np.ndarray:
        """Sync: (seq_len, 54, 2) pixel -> probabilities (num_classes,). spoter_v3.onnx
        already outputs softmax, so the result is used as-is.
        """
        tensor = self._transforms(torch.from_numpy(snapshot))
        poses = tensor.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, seq_len, 54, 2)
        return self.session.run(["probabilities"], {"poses": poses})[0][0]

    def _infer_sync(self, snapshot: np.ndarray) -> tuple[int, float]:
        probs = self._probs_sync(snapshot)
        idx = int(probs.argmax())
        return idx, float(probs[idx])

    def _fused_infer_sync(
        self, spoter_snapshot: np.ndarray, slgcn_snapshot: np.ndarray | None
    ) -> tuple[int, float, bool]:
        """Late-fusion SPOTER + SL-GCN for realtime/REST; falls back to SPOTER-only if
        SL-GCN is unavailable. Returns (idx, confidence, fused?).
        """
        p_spoter = self._probs_sync(spoter_snapshot)
        # Lazy import to avoid a circular import with dependencies.py.
        from serving.dependencies import get_slgcn
        from serving.services.slgcn import preprocess_snapshot

        slgcn = get_slgcn()
        if slgcn_snapshot is not None and slgcn.session is not None:
            p_slgcn = slgcn.probs(preprocess_snapshot(slgcn_snapshot))
            w = settings.fusion_w_slgcn
            probs = w * p_slgcn + (1.0 - w) * p_spoter
            fused = True
        else:
            probs = p_spoter
            fused = False
        idx = int(probs.argmax())
        return idx, float(probs[idx]), fused

    async def infer(self, snapshot: np.ndarray) -> tuple[int, float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._infer_sync, snapshot)

    async def fused_infer(
        self, spoter_snapshot: np.ndarray, slgcn_snapshot: np.ndarray | None
    ) -> tuple[int, float, bool]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, self._fused_infer_sync, spoter_snapshot, slgcn_snapshot
        )
