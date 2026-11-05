"""Sign recognition from a video FILE: reads frames, segments by arm-angle heuristic,
classifies each segment. Logic ported from demo_gradio.py (not imported: it has
import-time side effects); reuses SpoterService's ONNX session (ThreadPoolExecutor)."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np

from serving.config import settings
from serving.recognition_bridge import (
    load_pose_extract,
    load_slgcn_transforms,
    load_spoter_graph_transforms,
)
from serving.services.gloss_decoder import dedupe_consecutive
from serving.utils.logger import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)
_mp_pose = mp.solutions.pose


def _arm_angle(shoulder, elbow, wrist):
    """Elbow angle (shoulder-elbow-wrist), degrees; ~180 straight, smaller when bent."""
    radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(
        shoulder[1] - elbow[1], shoulder[0] - elbow[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def _hands_up_from_pose(pose, frame_rgb, angle_threshold, visibility_threshold):
    """(hand up?, visible wrist coords [normalized 0..1]). "Up" = at least one arm has
    elbow angle below threshold and a visible wrist."""
    res = pose.process(frame_rgb)
    if not res.pose_landmarks:
        return False, []
    lm = res.pose_landmarks.landmark
    P = _mp_pose.PoseLandmark
    up = False
    wrists = []
    for side in ("LEFT", "RIGHT"):
        sh = lm[getattr(P, f"{side}_SHOULDER").value]
        el = lm[getattr(P, f"{side}_ELBOW").value]
        wr = lm[getattr(P, f"{side}_WRIST").value]
        if wr.visibility < visibility_threshold:
            continue
        wrists.append((wr.x, wr.y))
        if _arm_angle((sh.x, sh.y), (el.x, el.y), (wr.x, wr.y)) < angle_threshold:
            up = True
    return up, wrists


def _track_motion(track):
    """Largest displacement (normalized coords) across the wrist position track."""
    pts = np.array([p for p in track if p is not None], dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.ptp(pts, axis=0).max())


def read_frames(path: str):
    """Reads a video file -> (RGB frames, fps, width, height); errors if none read."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError("Khong doc duoc khung hinh nao tu video.")
    h, w = frames[0].shape[:2]
    return frames, float(fps), int(w), int(h)


def segment_video(
    frames,
    angle_threshold=150.0,
    visibility_threshold=0.6,
    min_up_frames=3,
    min_down_frames=8,
    min_sign_frames=8,
    max_sign_frames=90,
    preroll_frames=5,
    min_wrist_motion=0.04,
    model_complexity=None,
):
    """Splits a multi-sign video into frame segments (~1 sign each); offline version of
    the realtime hand up/down logic. Segments with near-still hands (< min_wrist_motion)
    are dropped as a resting pose; model_complexity defaults from config (0=fastest)."""
    if model_complexity is None:
        model_complexity = settings.video_pose_model_complexity
    pose = _mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
        model_complexity=model_complexity,
    )
    segments, active, up, down, buf, track = [], False, 0, 0, [], []
    preroll = deque(maxlen=preroll_frames)
    try:
        for fr in frames:
            hands_up, wrists = _hands_up_from_pose(
                pose, fr, angle_threshold, visibility_threshold
            )
            wrist = tuple(np.mean(wrists, axis=0)) if wrists else None
            if not active:
                preroll.append(fr)
                if hands_up:
                    up += 1
                    if up >= min_up_frames:
                        active, buf, down, track = True, list(preroll), 0, []
                else:
                    up = 0
            else:
                buf.append(fr)
                track.append(wrist)
                down = 0 if hands_up else down + 1
                if down >= min_down_frames or len(buf) >= max_sign_frames:
                    if len(buf) >= min_sign_frames and _track_motion(track) >= min_wrist_motion:
                        segments.append(buf)
                    active, up, down, buf, track = False, 0, 0, [], []
                    preroll.clear()
        # Trailing segment at end of video (hands never came down).
        if active and len(buf) >= min_sign_frames and _track_motion(track) >= min_wrist_motion:
            segments.append(buf)
    finally:
        pose.close()
    return segments


class VideoRecognizerService:
    def __init__(self):
        self._pose_extract = None
        self._spoter_transforms = None
        self._slgcn_transforms = None

    def load(self) -> None:
        """Loads late-fusion preprocessing: shared PoseExtract + SPOTER/SL-GCN branches;
        reuses the ONNX sessions already loaded by SpoterService + SlgcnService."""
        self._pose_extract = load_pose_extract()
        self._spoter_transforms = load_spoter_graph_transforms()
        self._slgcn_transforms = load_slgcn_transforms()
        logger.info(
            "VideoRecognizer san sang — fusion SPOTER + SL-GCN (W_SLGCN=%.2f), "
            "dung lai phien ONNX cua SpoterService/SlgcnService.",
            settings.fusion_w_slgcn,
        )

    def _classify_frames(self, frames, fps, w, h) -> tuple[str | None, float]:
        """RGB frames -> (top-1 gloss | None, confidence) via SPOTER+SL-GCN late-fusion:
        extracts Pose once, branches, merges as P = W*P_slgcn + (1-W)*P_spoter.
        Both ONNX graphs already output softmaxed probabilities."""
        # Lazy import: dependencies.py imports this module (would be circular).
        from serving.dependencies import get_gloss_decoder, get_slgcn, get_spoter

        spoter_session = get_spoter().session
        if spoter_session is None:
            raise RuntimeError("SpoterService chua nap — khong co phien ONNX de infer.")
        slgcn = get_slgcn()

        # Shared PoseExtract (runs once); lower Holistic complexity for speed.
        # NOTE: don't trim frames before normalization -- it uses whole-sequence stats.
        holistic_config = {"model_complexity": settings.video_holistic_model_complexity}
        pose = self._pose_extract(
            {"frames": frames, "fps": fps, "width": w, "height": h, "holistic_config": holistic_config}
        )

        # SPOTER branch first (reads raw pixels only, doesn't mutate Pose).
        spoter_tensor = self._spoter_transforms(pose)
        spoter_in = spoter_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, 70, 54, 2)
        p_spoter = spoter_session.run(["probabilities"], {"poses": spoter_in})[0][0]

        # SL-GCN branch after (calls normalize_distribution, mutating Pose).
        # Falls back to SPOTER-only if SL-GCN isn't loaded (keeps the video flow alive).
        if slgcn.session is not None:
            slgcn_arr = self._slgcn_transforms(pose)                        # (3, 150, 27, 1)
            slgcn_in = slgcn_arr[None].astype(np.float32)                   # (1, 3, 150, 27, 1)
            p_slgcn = slgcn.probs(slgcn_in)
            w_slgcn = settings.fusion_w_slgcn
            probs = w_slgcn * p_slgcn + (1.0 - w_slgcn) * p_spoter
        else:
            logger.warning("SL-GCN chua san sang — luong video tam thoi SPOTER-only.")
            probs = p_spoter

        idx = int(probs.argmax())
        gloss = get_gloss_decoder().label(idx)
        return gloss, float(probs[idx])

    def recognize_video_file(self, path: str) -> list[str]:
        """Sync (runs in the thread pool): video file -> ordered gloss list. Segments
        the video into signs, classifies each, keeps only confident glosses."""
        if self._pose_extract is None:
            self.load()
        frames, fps, w, h = read_frames(path)
        segments = segment_video(frames)
        glosses: list[str] = []
        skipped = 0
        for seg_frames in segments:
            gloss, conf = self._classify_frames(seg_frames, fps, w, h)
            if gloss is None or conf < settings.video_confidence_threshold:
                skipped += 1
                continue
            glosses.append(gloss)
        # A video's consecutive duplicate words (a sign split across segments) collapse.
        deduped = dedupe_consecutive(glosses)
        logger.info(
            "Video: %d doan cat, %d tu nhan dang (%d sau khi bo trung lien ke), %d doan bo (duoi %.0f%%).",
            len(segments),
            len(glosses),
            len(deduped),
            skipped,
            settings.video_confidence_threshold * 100,
        )
        return deduped

    async def recognize(self, path: str) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self.recognize_video_file, path)
