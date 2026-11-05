"""SL-GCN ensemble (ONNX) wrapper, the 2nd late-fusion branch. Input is preprocessed to
(1, 3, slgcn_seq_len, num_points, 1); the ONNX graph bakes in bone/motion/normalize/
softmax-avg, so its "probabilities" output is used as-is."""

import numpy as np
import onnxruntime as ort

from serving.config import settings
from serving.utils.logger import get_logger

logger = get_logger(__name__)


def _pad(data: np.ndarray, num_frames: int) -> np.ndarray:
    """(T, V, 3) -> (3, num_frames, V, 1); repeat if short, crop if long (SLGCNPad)."""
    V, C = data.shape[1], data.shape[2]
    padded = np.zeros((num_frames, V, C, 1), dtype=np.float32)
    L = data.shape[0]
    if L == 0:
        return np.transpose(padded, [2, 0, 1, 3])
    if L < num_frames:
        padded[:L, :, :, 0] = data
        rest = num_frames - L
        num = int(np.ceil(rest / L))
        pad = np.concatenate([data for _ in range(num)], 0)[:rest]
        padded[L:, :, :, 0] = pad
    else:
        padded[:, :, :, 0] = data[:num_frames, :, :]
    return np.transpose(padded, [2, 0, 1, 3])


def preprocess_snapshot(snapshot: np.ndarray) -> np.ndarray:
    """(T, 27, 3) raw [x,y,confidence] -> (1, 3, slgcn_seq_len, 27, 1). Reproduces
    normalize_distribution + pad from the AI core; verified bit-exact on real video.
    """
    snap = np.asarray(snapshot, dtype=np.float64)
    T, V, _ = snap.shape
    norm = np.zeros((T, V, 3), dtype=np.float64)
    conf = snap[:, :, 2]
    valid = conf > 0
    for ch in (0, 1):
        for k in range(V):
            m = valid[:, k]
            vals = snap[m, k, ch]
            if vals.size:
                std = vals.std()
                if std > 0:
                    z = (snap[:, k, ch] - vals.mean()) / std
                    norm[m, k, ch] = z[m]  # only frames with confidence; others stay 0
    norm[:, :, 2] = conf
    return _pad(norm, settings.slgcn_seq_len)[None].astype(np.float32)


class SlgcnService:
    def __init__(self):
        self.session: ort.InferenceSession | None = None

    def load(self) -> None:
        # Helps onnxruntime-gpu find CUDA/cuDNN installed via pip (nvidia-*-cu12).
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                logger.warning("preload_dlls that bai, bo qua.")
        available = ort.get_available_providers()
        providers = [p for p in settings.spoter_providers if p in available]
        self.session = ort.InferenceSession(str(settings.slgcn_onnx_path), providers=providers)
        logger.info("SL-GCN san sang. ONNX providers: %s", self.session.get_providers())

    def probs(self, slgcn_input: np.ndarray) -> np.ndarray:
        """(1, 3, slgcn_seq_len, num_points, 1) -> probabilities (num_classes,)."""
        if self.session is None:
            raise RuntimeError("SlgcnService chua nap — khong co phien ONNX de infer.")
        return self.session.run(["probabilities"], {"poses": slgcn_input})[0][0]
