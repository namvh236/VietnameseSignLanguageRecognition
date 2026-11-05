"""Run the real KeypointSegmenter over eval videos and dump segment predictions to JSON."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

from serving.config import settings  # noqa: E402
from serving.services.segmenter import KeypointSegmenter  # noqa: E402
from serving.utils.logger import get_logger  # noqa: E402

logger = get_logger("eval_segmentation")

# SPOTER 54-keypoint order; must mirror frontend/src/utils/keypoints.js.
# None = synthesized neck (MediaPipe does not provide one).
POSE_MAP = [0, None, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15]
# MediaPipe Hand landmark indices in JointSelect order (shared by both hands).
HAND_MAP = [0, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 19, 20, 18, 17, 4, 3, 2, 1]

# Holistic params must match frontend/src/services/holistic.js
# (thresholds are tuned to this landmark layout).
HOLISTIC_OPTIONS = dict(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    refine_face_landmarks=False,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)


def extract_spoter_keypoints(results, width: float, height: float) -> np.ndarray:
    out = np.zeros((54, 2), dtype=np.float32)
    o = 0

    pose = results.pose_landmarks.landmark if results.pose_landmarks else None
    for idx in POSE_MAP:
        if idx is not None and pose is not None:
            lm = pose[idx]
            out[o, 0] = lm.x * width
            out[o, 1] = lm.y * height
        o += 1

    for hand_landmarks in (results.left_hand_landmarks, results.right_hand_landmarks):
        hand = hand_landmarks.landmark if hand_landmarks else None
        for idx in HAND_MAP:
            if hand is not None:
                lm = hand[idx]
                out[o, 0] = lm.x * width
                out[o, 1] = lm.y * height
            o += 1

    return out


def make_segmenter(args: argparse.Namespace) -> KeypointSegmenter:
    return KeypointSegmenter(
        angle_threshold=args.seg_angle_threshold,
        min_up_frames=args.seg_min_up_frames,
        min_down_frames=args.seg_min_down_frames,
        min_sign_frames=args.seg_min_sign_frames,
        max_sign_frames=args.seg_max_sign_frames,
        preroll_frames=args.seg_preroll_frames,
        min_wrist_motion_px=args.seg_min_wrist_motion_px,
        min_hand_frames_ratio=settings.min_hand_frames_ratio,
    )


def run_video(path: Path, args: argparse.Namespace) -> list[dict]:
    segmenter = make_segmenter(args)
    segments: list[dict] = []

    cap = cv2.VideoCapture(str(path))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0

    frame_idx = -1
    with mp.solutions.holistic.Holistic(**HOLISTIC_OPTIONS) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            spoter_kp = extract_spoter_keypoints(results, width, height)
            clip = segmenter.push(spoter_kp, None)
            if clip is not None:
                spoter_clip, _ = clip
                length = spoter_clip.shape[0]
                end_frame = frame_idx
                start_frame = end_frame - length + 1
                segments.append({"start_frame": start_frame, "end_frame": end_frame})

        pending = segmenter.flush()
        if pending is not None:
            spoter_clip, _ = pending
            length = spoter_clip.shape[0]
            end_frame = frame_idx
            start_frame = end_frame - length + 1
            segments.append({"start_frame": start_frame, "end_frame": end_frame})

    cap.release()
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", help="Thu muc segmentation_eval_video (chua manifest.json + videos/)")
    parser.add_argument("--out", default="predictions.json", help="File JSON ket qua")
    parser.add_argument("--seg-angle-threshold", type=float, default=settings.seg_angle_threshold)
    parser.add_argument("--seg-min-up-frames", type=int, default=settings.seg_min_up_frames)
    parser.add_argument("--seg-min-down-frames", type=int, default=settings.seg_min_down_frames)
    parser.add_argument("--seg-min-sign-frames", type=int, default=settings.seg_min_sign_frames)
    parser.add_argument("--seg-max-sign-frames", type=int, default=settings.seg_max_sign_frames)
    parser.add_argument("--seg-preroll-frames", type=int, default=settings.seg_preroll_frames)
    parser.add_argument("--seg-min-wrist-motion-px", type=float, default=settings.seg_min_wrist_motion_px)
    parser.add_argument("--notes", default="", help="Ghi chu tu do vao output")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    manifest = json.loads((eval_dir / "manifest.json").read_text(encoding="utf-8"))

    result = {
        "params": {
            "angle_threshold": args.seg_angle_threshold,
            "min_up_frames": args.seg_min_up_frames,
            "min_down_frames": args.seg_min_down_frames,
            "min_sign_frames": args.seg_min_sign_frames,
            "max_sign_frames": args.seg_max_sign_frames,
            "preroll_frames": args.seg_preroll_frames,
            "min_wrist_motion_px": args.seg_min_wrist_motion_px,
        },
        "notes": (
            f"mediapipe {mp.__version__}, Holistic python (model_complexity=1, "
            f"smooth_landmarks=True, min_detection/tracking_confidence=0.4), offline decode "
            f"bang OpenCV, nap tuan tu vao KeypointSegmenter that (khong gia lap WS). "
            + args.notes
        ),
        "videos": [],
    }

    videos = manifest["videos"]
    for i, entry in enumerate(videos):
        video_id = entry["video_id"]
        video_path = eval_dir / entry["path"]
        logger.info("[%d/%d] %s ...", i + 1, len(videos), video_id)
        segments = run_video(video_path, args)
        logger.info("  -> %d doan.", len(segments))
        result["videos"].append({"video_id": video_id, "predicted_segments": segments})

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Da ghi %s", out_path)


if __name__ == "__main__":
    main()
