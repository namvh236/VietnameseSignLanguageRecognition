"""Segments a realtime keypoint stream into one-sign clips (ported from demo_gradio.py's
RealtimeSegmenter, but on keypoint vectors, not frames); elbow angle marks hand up/down.
Model is isolated-sign, so signers must drop hands between words."""

from collections import deque

import numpy as np

from serving.models.schemas import NUM_JOINTS, SLGCN_NUM_JOINTS
from serving.utils.logger import get_logger

logger = get_logger(__name__)

# Body joint indices in the 54-joint SPOTER layout (JointSelect order):
#   [nose, neck, rightEye, leftEye, rightEar, leftEar,
#    rightShoulder(6), leftShoulder(7), rightElbow(8), leftElbow(9), rightWrist(10), leftWrist(11)]
_SIDES = {
    "LEFT": (7, 9, 11),   # (shoulder, elbow, wrist)
    "RIGHT": (6, 8, 10),
}

# Each hand's 21-joint block in the 54-joint layout (body, left hand, right hand).
_HAND_SLICES = (slice(12, 33), slice(33, 54))


def _hand_present(spoter_kp: np.ndarray) -> bool:
    """At least one hand detected (MediaPipe Hands) this frame? Undetected hand joints
    are [0, 0] (see frontend/src/utils/keypoints.js); the main anti-false-positive
    filter, since Pose still guesses wrist/elbow off-camera but Hands does not."""
    return any(bool(np.any(spoter_kp[s])) for s in _HAND_SLICES)


def _arm_angle(shoulder, elbow, wrist) -> float:
    """Elbow angle (shoulder-elbow-wrist), degrees; ~180 straight, smaller when bent."""
    radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(
        shoulder[1] - elbow[1], shoulder[0] - elbow[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def _track_motion(track) -> float:
    """Largest pixel displacement (range) across the wrist position track."""
    pts = np.array([p for p in track if p is not None], dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.ptp(pts, axis=0).max())


class KeypointSegmenter:
    """Per-frame state machine: splits a keypoint stream into clips via hand up/down."""

    def __init__(
        self,
        angle_threshold: float,
        min_up_frames: int,
        min_down_frames: int,
        min_sign_frames: int,
        max_sign_frames: int,
        preroll_frames: int,
        min_wrist_motion_px: float,
        min_hand_frames_ratio: float,
    ):
        self.angle_threshold = angle_threshold
        self.min_up_frames = min_up_frames
        self.min_down_frames = min_down_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.min_wrist_motion_px = min_wrist_motion_px
        self.min_hand_frames_ratio = min_hand_frames_ratio
        self._preroll_spoter: deque = deque(maxlen=preroll_frames)
        self._preroll_slgcn: deque = deque(maxlen=preroll_frames)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.up_count = 0
        self.down_count = 0
        self._spoter_buf: list[np.ndarray] = []  # each element is (54, 2)
        self._slgcn_buf: list[np.ndarray | None] = []  # each element is (27, 3) or None
        self._wrist_track: list = []
        self._preroll_spoter.clear()
        self._preroll_slgcn.clear()

    def _hands_up(self, spoter_kp: np.ndarray):
        """(hand up?, visible wrists); spoter_kp: (54, 2) pixel. "Up" = elbow angle
        below threshold AND a hand detected -- without the hand check, resting posture
        (elbow bent ~90-140) is misread as signing."""
        if not _hand_present(spoter_kp):
            return False, []
        up = False
        wrists = []
        for shoulder_i, elbow_i, wrist_i in _SIDES.values():
            wr = spoter_kp[wrist_i]
            if wr[0] == 0 and wr[1] == 0:  # wrist not detected -> joint is [0, 0]
                continue
            wrists.append((float(wr[0]), float(wr[1])))
            if _arm_angle(spoter_kp[shoulder_i], spoter_kp[elbow_i], wr) < self.angle_threshold:
                up = True
        return up, wrists

    def push(
        self, spoter_frame: np.ndarray, slgcn_frame: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Push one keypoint frame. Returns (SPOTER clip, SL-GCN clip|None) once a sign
        just ended, else None."""
        sp = np.asarray(spoter_frame, dtype=np.float32).reshape(NUM_JOINTS, 2)
        sg = (
            np.asarray(slgcn_frame, dtype=np.float32).reshape(SLGCN_NUM_JOINTS, 3)
            if slgcn_frame is not None
            else None
        )
        up, wrists = self._hands_up(sp)
        wrist = tuple(np.mean(wrists, axis=0)) if wrists else None

        if not self.active:
            self._preroll_spoter.append(sp)
            self._preroll_slgcn.append(sg)
            if up:
                self.up_count += 1
                if self.up_count >= self.min_up_frames:
                    # Start of a sign: seed buffer with pre-roll to keep the lead-in.
                    self.active = True
                    self._spoter_buf = list(self._preroll_spoter)
                    self._slgcn_buf = list(self._preroll_slgcn)
                    self._wrist_track = [wrist]
                    self.down_count = 0
            else:
                self.up_count = 0
            return None

        # Mid-sign: keep accumulating.
        self._spoter_buf.append(sp)
        self._slgcn_buf.append(sg)
        self._wrist_track.append(wrist)
        self.down_count = 0 if up else self.down_count + 1

        ended_by_hands_down = self.down_count >= self.min_down_frames
        overflowed = len(self._spoter_buf) >= self.max_sign_frames
        if not (ended_by_hands_down or overflowed):
            return None

        spoter_buf, slgcn_buf, track = self._spoter_buf, self._slgcn_buf, self._wrist_track
        # Reset to idle, ready for the next sign.
        self.active = False
        self.up_count = 0
        self.down_count = 0
        self._spoter_buf = []
        self._slgcn_buf = []
        self._wrist_track = []
        self._preroll_spoter.clear()
        self._preroll_slgcn.clear()

        if not ended_by_hands_down:
            # Overflowed without ever seeing hands go down: almost certainly a stale
            # frame, not a real boundary. Drop the clip instead of guessing a label.
            logger.debug("Bo clip tran %d frame ma khong thay ha tay.", len(spoter_buf))
            return None

        return self._finalize_clip(spoter_buf, slgcn_buf, track)

    def flush(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Finalizes the in-progress clip, if any (session end / client end_sign)."""
        if not self.active:
            return None
        spoter_buf, slgcn_buf, track = self._spoter_buf, self._slgcn_buf, self._wrist_track
        self.reset()
        return self._finalize_clip(spoter_buf, slgcn_buf, track)

    def _finalize_clip(self, spoter_buf, slgcn_buf, track):
        if len(spoter_buf) < self.min_sign_frames:
            logger.debug("Bo clip qua ngan (%d frame).", len(spoter_buf))
            return None  # too short -> noise
        hand_ratio = sum(_hand_present(f) for f in spoter_buf) / len(spoter_buf)
        if hand_ratio < self.min_hand_frames_ratio:
            # Preroll frames may lack a hand; only drop when the whole clip does.
            logger.debug("Bo clip thieu ban tay (ti le %.2f).", hand_ratio)
            return None
        motion = _track_motion(track)
        if motion < self.min_wrist_motion_px:
            logger.debug("Bo clip tay gan nhu dung yen (%.1f px).", motion)
            return None  # wrist barely moved -> resting posture, not a sign
        spoter_clip = np.stack(spoter_buf).astype(np.float32)  # (T, 54, 2)
        # Only fuse when the whole clip has SL-GCN frames (keeps them time-aligned).
        if all(f is not None for f in slgcn_buf):
            slgcn_clip = np.stack(slgcn_buf).astype(np.float32)  # (T, 27, 3)
        else:
            slgcn_clip = None
        return spoter_clip, slgcn_clip
