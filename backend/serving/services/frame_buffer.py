"""Sliding window buffer for keypoint frames (one per session); keeps parallel SPOTER
and SL-GCN frames for realtime late-fusion. SL-GCN is optional (falls back to
SPOTER-only)."""

from collections import deque

import numpy as np

from serving.models.schemas import NUM_JOINTS, SLGCN_NUM_JOINTS


class FrameBuffer:
    def __init__(self, seq_len: int, stride: int):
        self.seq_len = seq_len
        self.stride = stride
        self._buf: deque[np.ndarray] = deque(maxlen=seq_len)
        self._slgcn_buf: deque[np.ndarray | None] = deque(maxlen=seq_len)
        self._frames_since_last = 0

    def push(
        self, frame: np.ndarray, slgcn_frame: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Push one SPOTER frame (+ optional SL-GCN frame). Returns (SPOTER snapshot,
        SL-GCN snapshot | None) once the window is full and enough stride frames have
        accumulated; SL-GCN snapshot is None if the client sent no SL-GCN frames."""
        self._buf.append(frame.reshape(NUM_JOINTS, 2))
        self._slgcn_buf.append(
            slgcn_frame.reshape(SLGCN_NUM_JOINTS, 3) if slgcn_frame is not None else None
        )
        self._frames_since_last += 1

        if len(self._buf) == self.seq_len and self._frames_since_last >= self.stride:
            self._frames_since_last = 0
            spoter_snap = np.array(self._buf)
            slgcn_frames = [f for f in self._slgcn_buf if f is not None]
            # Only fuse when the whole window has SL-GCN frames (keeps them aligned).
            slgcn_snap = (
                np.array(slgcn_frames) if len(slgcn_frames) == len(self._buf) else None
            )
            return spoter_snap, slgcn_snap
        return None

    def reset(self) -> None:
        self._buf.clear()
        self._slgcn_buf.clear()
        self._frames_since_last = 0

    @property
    def fill_ratio(self) -> float:
        return len(self._buf) / self.seq_len
