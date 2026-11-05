"""Gradio demo: sign video -> gloss (fusion) -> sentence (LLM) -> speech (TTS)."""

import os
import re
import sys
import csv
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

import onnxruntime as ort
import gradio as gr
import mediapipe as mp
from torchvision.transforms.v2 import Compose

from pose_format import Pose

from pipelines.spoter_graph_classification import (
    PoseExtract,
    JointSelect,
    TensorToDict,
    SingleBodyDictNormalize,
    SPOTERSingleHandDictNormalize,
    DictToTensor,
    Shift,
    Pad,
)
from utils import SLGCN_JOINTS, COCO_TO_POSE_FORMAT

# Late fusion: P_fused = W_SLGCN * P_slgcn + (1 - W_SLGCN) * P_spoter -> argmax.
# Both ONNX graphs already output softmaxed probabilities; do not softmax again.
SPOTER_ONNX_PATH = ROOT / "models" / "spoter" / "spoter_v3.onnx"
SLGCN_ONNX_PATH = ROOT / "models" / "sl-gcn" / "sl_gcn_ensemble.onnx"
GLOSS_CSV = ROOT / "experiments" / "gloss.csv"
# Weights tuned on val/test: SL-GCN 0.75 / SPOTER 0.25 (fusion 90.1%, beats solo).
W_SLGCN = float(os.getenv("FUSION_W_SLGCN", "0.75"))
SPOTER_NUM_FRAMES = 70
SLGCN_NUM_FRAMES = 150
SLGCN_NUM_POINTS = 27
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
# Top-1 confidence threshold to auto-accept a word (realtime / multi-sign video).
# The 400-class model always returns a class even for garbage; filter by confidence.
MIN_CONFIDENCE = float(os.getenv("SIGN_MIN_CONF", "0.5"))


# Copied from src/features/transforms/sl_gcn.py, intentionally dropping the
# `_build_edges` import: bone/motion/normalize are already baked into the ONNX graph.

class SLGCNJointSelect:
    """Select SL-GCN joints, each as (x, y, confidence)."""

    def __init__(self, num_points: int = SLGCN_NUM_POINTS) -> None:
        self.joints = SLGCN_JOINTS[num_points]

    def __get_point(self, component: str, point: str, pose: Pose) -> np.ndarray:
        idx = pose.header._get_point_index(component, point)
        T, _, _, C = pose.body.data.shape
        data = np.zeros((T, C), dtype=pose.body.data.dtype)
        data[:, :2] = pose.body.data[:, 0, idx, :2].data
        data[:, 2] = pose.body.confidence[:, 0, idx]
        return data

    def __call__(self, pose: Pose) -> np.ndarray:
        pose.normalize_distribution()
        data = []
        for joint in self.joints:
            component, point = COCO_TO_POSE_FORMAT[joint]
            data.append(self.__get_point(component, point, pose))
        # (num_landmarks, num_frames, 3) -> (num_frames, num_landmarks, 3)
        return np.array(data).transpose((1, 0, 2))


class SLGCNPad:
    """Pad/crop to num_frames and transpose to (C, T, V, M) for SL-GCN ONNX."""

    def __init__(self, num_frames: int = SLGCN_NUM_FRAMES) -> None:
        self.num_frames = num_frames

    def __call__(self, data: np.ndarray) -> np.ndarray:
        padded_data = np.zeros(
            (self.num_frames, data.shape[1], data.shape[2], 1),
            dtype=np.float32,
        )
        L = data.shape[0]
        if L < self.num_frames:
            padded_data[:L, :, :, 0] = data
            rest = self.num_frames - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([data for _ in range(num)], 0)[:rest]
            padded_data[L:, :, :, 0] = pad
        else:
            padded_data[:, :, :, 0] = data[:self.num_frames, :, :]
        # (num_frames, num_points, num_channels, num_people)
        # -> (num_channels, num_frames, num_points, num_people)
        return np.transpose(padded_data, [2, 0, 1, 3])


def read_labels(path: Path) -> dict[int, str]:
    id2label: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                id2label[int(row[0])] = row[1]
    return id2label


print("Đang nạp 2 model nhận dạng (SPOTER + SL-GCN) + nhãn...")
ID2LABEL = read_labels(GLOSS_CSV)
# Helps onnxruntime-gpu find pip-installed CUDA/cuDNN (nvidia-*-cu12 packages).
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls()
    except Exception:  # noqa: BLE001
        pass
# Prefer CUDA for ONNX, falling back to CPU if unavailable.
_available = ort.get_available_providers()
_providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in _available]
SPOTER_SESSION = ort.InferenceSession(str(SPOTER_ONNX_PATH), providers=_providers)
SLGCN_SESSION = ort.InferenceSession(str(SLGCN_ONNX_PATH), providers=_providers)
print(f"ONNX providers — SPOTER: {SPOTER_SESSION.get_providers()} | SL-GCN: {SLGCN_SESSION.get_providers()}")

POSE_EXTRACT = PoseExtract()
# SPOTER branch: Pose -> tensor (70, 54, 2).
SPOTER_TRANSFORMS = Compose(
    [
        JointSelect(),
        TensorToDict(),
        SingleBodyDictNormalize(),
        SPOTERSingleHandDictNormalize(),
        DictToTensor(),
        Shift(),
        Pad(SPOTER_NUM_FRAMES),
    ]
)
# SL-GCN branch: Pose -> array (3, 150, 27, 1).
SLGCN_TRANSFORMS = Compose(
    [
        SLGCNJointSelect(SLGCN_NUM_POINTS),
        SLGCNPad(SLGCN_NUM_FRAMES),
    ]
)
print(f"Sẵn sàng — {len(ID2LABEL)} lớp gloss | fusion W_SLGCN={W_SLGCN:.2f}.")

# LLM/TTS are lazy-loaded (heavy); created on first use only.
_llm_client = None
_tts_engine = None


def _get_llm():
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI

        _llm_client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    return _llm_client


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        from vieneu import Vieneu

        print("Đang nạp VieNeu-TTS (lần đầu có thể lâu)...")
        _tts_engine = Vieneu(mode="standard")
    return _tts_engine


def read_frames(path: str):
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
        raise gr.Error("Không đọc được khung hình nào từ video.")
    h, w = frames[0].shape[:2]
    return frames, float(fps), int(w), int(h)


def classify_frames(frames, fps, w, h):
    """Frames -> (top-5 {gloss: prob}, top-1 gloss) via SPOTER+SL-GCN fusion.
    Both ONNX graphs output pre-softmaxed probabilities; don't softmax again.
    """
    pose = POSE_EXTRACT({"frames": frames, "fps": fps, "width": w, "height": h})

    # SPOTER first: it only reads pose data, doesn't mutate it.
    spoter_tensor = SPOTER_TRANSFORMS(pose)
    spoter_in = spoter_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)  # (1, 70, 54, 2)
    p_spoter = SPOTER_SESSION.run(["probabilities"], {"poses": spoter_in})[0][0]

    # SL-GCN second: SLGCNJointSelect calls normalize_distribution, mutating Pose.
    slgcn_arr = SLGCN_TRANSFORMS(pose)                                       # (3, 150, 27, 1)
    slgcn_in = slgcn_arr[None].astype(np.float32)                           # (1, 3, 150, 27, 1)
    p_slgcn = SLGCN_SESSION.run(["probabilities"], {"poses": slgcn_in})[0][0]

    probs = W_SLGCN * p_slgcn + (1.0 - W_SLGCN) * p_spoter
    top5 = np.argsort(probs)[::-1][:5]
    scores = {ID2LABEL[int(i)]: float(probs[i]) for i in top5}
    top_gloss = ID2LABEL[int(top5[0])]
    return scores, top_gloss


def recognize(video_path, signs, multi):
    """Video -> gloss(es). If `multi`, auto-split by hand up/down into several signs;
    otherwise treat the whole video as one sign."""
    if not video_path:
        raise gr.Error("Hãy quay hoặc tải lên một video ký hiệu.")
    frames, fps, w, h = read_frames(video_path)

    if multi:
        segments = segment_video(frames)
        if not segments:
            raise gr.Error(
                "Không tách được ký hiệu nào. Hãy hạ tay xuống giữa các từ, hoặc bỏ chọn "
                "'Video có nhiều từ' nếu video chỉ có một ký hiệu."
            )
        new_signs = list(signs)
        scores = {}
        skipped = 0
        for seg_frames in segments:
            scores, top_gloss = classify_frames(seg_frames, fps, w, h)
            # Auto-cut segments may be junk; only accept if confident enough.
            if scores.get(top_gloss, 0.0) < MIN_CONFIDENCE:
                skipped += 1
                continue
            new_signs.append(top_gloss)
        if skipped:
            gr.Warning(f"Đã bỏ qua {skipped} đoạn có độ tin cậy dưới {MIN_CONFIDENCE:.0%}.")
        return scores, new_signs, "  •  ".join(new_signs)

    scores, top_gloss = classify_frames(frames, fps, w, h)
    new_signs = signs + [top_gloss]
    return scores, new_signs, "  •  ".join(new_signs)


_CONTROL_TOKEN_RE = re.compile(r"\s*/(?:no_)?think\b")
_LABEL_ONLY_LINE_RE = re.compile(
    r"^(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：.\-–]*\s*$",
    flags=re.IGNORECASE,
)
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：]\s*",
    flags=re.IGNORECASE,
)


def _strip_think(text: str) -> str:
    """Strip <think> blocks, control tokens, and label lines Qwen3 sometimes leaks
    into its output (e.g. /no_think, or a label prefix like 'Bản dịch:')."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = _CONTROL_TOKEN_RE.sub("", text)
    cleaned: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _LABEL_ONLY_LINE_RE.match(line):
            continue
        line = _LABEL_PREFIX_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned).strip()


def _dedupe_consecutive(signs: list[str]) -> list[str]:
    """Drop consecutive duplicate signs (same gesture recognized multiple times)."""
    return [g for i, g in enumerate(signs) if i == 0 or g != signs[i - 1]]


def generate_sentence(signs: list[str]) -> str:
    system_prompt = (
        "Bạn là trợ lý dịch ngôn ngữ ký hiệu tiếng Việt. Đầu vào là danh sách các từ (gloss) "
        "nhận dạng được từ người ký hiệu, theo đúng thứ tự thể hiện. Nhiệm vụ của bạn: ghép "
        "chúng thành MỘT câu tiếng Việt tự nhiên, đúng ngữ pháp, dễ hiểu.\n"
        "QUY TẮC BẮT BUỘC:\n"
        "- Dùng ĐỦ và ĐÚNG tất cả các từ đã cho, giữ nguyên thứ tự xuất hiện.\n"
        "- KHÔNG thêm từ mang nghĩa mới, KHÔNG bịa nội dung, KHÔNG bỏ bớt từ nào.\n"
        "- Chỉ được thêm tối thiểu hư từ/từ nối (là, và, thì, của, đang, ạ, ơi, không...) "
        "và dấu câu khi thật cần thiết để câu đúng ngữ pháp.\n"
        "- Nếu chỉ có một từ, trả về đúng từ đó (viết hoa chữ đầu, thêm dấu câu).\n"
        "VÍ DỤ:\n"
        "- Từ: tôi, muốn, ăn, cơm → Tôi muốn ăn cơm.\n"
        "- Từ: mẹ, đi, chợ → Mẹ đi chợ.\n"
        "- Từ: anh, khỏe, không → Anh khỏe không?\n"
        "- Từ: xin chào → Xin chào.\n"
        "Chỉ trả về câu kết quả, không giải thích, "
        "không dùng thẻ <think> hay định dạng đặc biệt."
    )
    client = _get_llm()
    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Các từ nhận dạng được: {', '.join(signs)} /no_think"},
        ],
        temperature=0.3,
        max_tokens=128,  # short sentences; cap tokens for faster inference
    )
    return _strip_think(response.choices[0].message.content)


def make_sentence_and_speak(signs):
    """Signs -> natural sentence (LLM) -> speech (TTS)."""
    if not signs:
        raise gr.Error("Chưa có từ nào. Hãy nhận dạng ít nhất một ký hiệu trước.")

    # Collapse repeated detections before building the sentence.
    signs = _dedupe_consecutive(signs)

    # 1) Generate sentence (fall back to raw words if Ollama fails).
    try:
        sentence = generate_sentence(signs)
    except Exception as exc:  # noqa: BLE001
        sentence = " ".join(signs)
        gr.Warning(f"LLM lỗi (kiểm tra Ollama đã chạy + pull {OLLAMA_MODEL}?): {exc}. Tạm ghép từ thô.")

    # Empty LLM output (e.g. all inside <think>) falls back to raw words.
    if not sentence or not sentence.strip():
        sentence = " ".join(signs)

    # 2) Generate speech.
    try:
        engine = _get_tts()
        audio = np.asarray(engine.infer(sentence), dtype=np.float32).reshape(-1)
        if audio.size == 0:
            gr.Warning("TTS trả về audio rỗng — bỏ qua phần giọng nói.")
            return sentence, None
        return sentence, (engine.sample_rate, audio)
    except Exception as exc:  # noqa: BLE001
        gr.Warning(f"TTS lỗi: {exc}")
        return sentence, None


def clear_all():
    return [], "", "", None, None


_mp_pose = mp.solutions.pose


def _arm_angle(shoulder, elbow, wrist):
    """Elbow angle (shoulder-elbow-wrist) in degrees; ~180 when straight, less when
    raised/bent."""
    radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(
        shoulder[1] - elbow[1], shoulder[0] - elbow[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def _hands_up_from_pose(pose, frame_rgb, angle_threshold, visibility_threshold):
    """(hand raised?, visible wrist coords [0..1]). Raised = elbow angle below
    threshold with wrist visible; wrists also gauge motion so a resting hand
    isn't mistaken for a sign."""
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
    """Max movement range (normalized coords) across a wrist-position track."""
    pts = np.array([p for p in track if p is not None], dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.ptp(pts, axis=0).max())


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
):
    """Split a multi-sign video into per-sign segments; mirrors the realtime hand
    up/down logic. Near-still wrist segments are dropped as idle, not signs."""
    pose = _mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True
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
        # Trailing in-progress segment at end of video (hand never went down).
        if active and len(buf) >= min_sign_frames and _track_motion(track) >= min_wrist_motion:
            segments.append(buf)
    finally:
        pose.close()
    return segments


class RealtimeSegmenter:
    """Per-frame state machine detecting sign start/end (Temporal Boundary
    Localization: elbow angle + wrist visibility), debounced via min_up/min_down
    frames, with pre-roll so the sign start isn't cut off."""

    def __init__(
        self,
        angle_threshold=150.0,
        visibility_threshold=0.6,
        min_up_frames=3,
        min_down_frames=8,
        min_sign_frames=8,
        max_sign_frames=90,
        preroll_frames=5,
        min_wrist_motion=0.04,
    ):
        self.angle_threshold = angle_threshold
        self.visibility_threshold = visibility_threshold
        self.min_up_frames = min_up_frames
        self.min_down_frames = min_down_frames
        self.min_sign_frames = min_sign_frames
        self.max_sign_frames = max_sign_frames
        self.min_wrist_motion = min_wrist_motion
        self.pose = _mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True
        )
        self.preroll = deque(maxlen=preroll_frames)
        self.reset_state()

    def reset_state(self):
        self.active = False  # inside a sign?
        self.up_count = 0  # consecutive "hand up" frames
        self.down_count = 0  # consecutive "hand down" frames
        self.buffer = []  # frames of the current sign
        self.wrist_track = []  # per-frame wrist pos (for motion amplitude)
        self.buf_t0 = None  # first-frame timestamp (to estimate fps)
        self.buf_t1 = None
        self.last_sign_time = None  # timestamp of the last recognized sign
        self.spoken_upto = 0  # words already auto-spoken (avoid repeats)

    def close(self):
        try:
            self.pose.close()
        except Exception:  # noqa: BLE001
            pass

    def _hands_up(self, frame_rgb):
        return _hands_up_from_pose(
            self.pose, frame_rgb, self.angle_threshold, self.visibility_threshold
        )

    def push(self, frame_rgb):
        """Push a frame; returns (frames, fps) when a sign just ended, else None."""
        now = time.time()
        up, wrists = self._hands_up(frame_rgb)
        wrist = tuple(np.mean(wrists, axis=0)) if wrists else None

        if not self.active:
            self.preroll.append(frame_rgb)
            if up:
                self.up_count += 1
                if self.up_count >= self.min_up_frames:
                    # Start of sign: seed the buffer with pre-roll to keep the start.
                    self.active = True
                    self.buffer = list(self.preroll)
                    self.wrist_track = [wrist]
                    self.buf_t0 = now
                    self.buf_t1 = now
                    self.down_count = 0
            else:
                self.up_count = 0
            return None

        self.buffer.append(frame_rgb)
        self.wrist_track.append(wrist)
        self.buf_t1 = now
        if up:
            self.down_count = 0
        else:
            self.down_count += 1

        finished = self.down_count >= self.min_down_frames or len(self.buffer) >= self.max_sign_frames
        if not finished:
            return None

        frames = self.buffer
        motion = _track_motion(self.wrist_track)
        dur = max(self.buf_t1 - self.buf_t0, 1e-3)
        fps = float(len(frames)) / dur if len(frames) > 1 else 25.0
        fps = float(np.clip(fps, 5.0, 60.0))
        # Reset to idle, ready for the next sign.
        self.active = False
        self.up_count = 0
        self.down_count = 0
        self.buffer = []
        self.wrist_track = []
        self.preroll.clear()
        if len(frames) < self.min_sign_frames:
            return None  # too short, likely noise
        if motion < self.min_wrist_motion:
            return None  # wrists barely moved: resting pose, not a sign
        return frames, fps


def realtime_reset():
    """Clear realtime mode state (including the segmenter)."""
    return [], "🟢 Sẵn sàng — hãy bắt đầu ký hiệu.", "", "", None, None


def realtime_step(frame, signs, segmenter, angle_thr, min_down, idle_secs):
    """Streaming handler: classify a sign when it ends; after idle_secs with new
    words, auto-build a sentence (LLM) and speak it (TTS)."""
    # Lazily create the segmenter, kept per session in gr.State.
    if segmenter is None:
        segmenter = RealtimeSegmenter()
    segmenter.angle_threshold = float(angle_thr)
    segmenter.min_down_frames = int(min_down)

    now = time.time()
    new_sign = False
    low_conf = None

    if frame is not None:
        result = segmenter.push(frame)
        if result is not None:
            frames, fps = result
            h, w = frame.shape[:2]
            try:
                scores, top_gloss = classify_frames(frames, fps, int(w), int(h))
                conf = scores.get(top_gloss, 0.0)
                if conf >= MIN_CONFIDENCE:
                    signs = signs + [top_gloss]
                    segmenter.last_sign_time = now
                    new_sign = True
                else:
                    # Low-confidence segment: skip instead of adding a wrong word.
                    low_conf = (top_gloss, conf)
            except Exception as exc:  # noqa: BLE001
                gr.Warning(f"Lỗi phân loại đoạn vừa cắt: {exc}")

    signs_text = "  •  ".join(signs)

    if segmenter.active:
        status = "✍️ Đang ký..."
    elif new_sign:
        status = f"✅ Nhận dạng: **{signs[-1]}**"
    elif low_conf:
        status = f"🤔 Bỏ qua đoạn không rõ (đoán *{low_conf[0]}* chỉ {low_conf[1]:.0%})."
    elif signs:
        status = "🟢 Nghỉ — chờ ký hiệu tiếp theo."
    else:
        status = "🟢 Sẵn sàng — hãy bắt đầu ký hiệu."

    # Auto-build + speak the sentence once idle long enough with unspoken words.
    pending = len(signs) > segmenter.spoken_upto
    idle_ok = (
        not segmenter.active
        and segmenter.last_sign_time is not None
        and (now - segmenter.last_sign_time) >= float(idle_secs)
    )
    if pending and idle_ok:
        segmenter.spoken_upto = len(signs)  # mark before speaking to avoid a retrigger
        sentence, audio = make_sentence_and_speak(signs)
        status = f"🗣️ Tự đọc câu: {sentence}"
        return signs, status, signs_text, segmenter, sentence, audio

    # Nothing new to speak; leave the sentence/audio outputs unchanged.
    return signs, status, signs_text, segmenter, gr.skip(), gr.skip()


def perword_step(frame, signs, segmenter, angle_thr, min_down):
    """Per-word tab handler: like realtime but never auto-speaks (idle disabled);
    sign one word, lower the hand to commit it, then repeat."""
    signs, status, signs_text, segmenter, _s, _a = realtime_step(
        frame, signs, segmenter, angle_thr, min_down, idle_secs=float("inf")
    )
    return signs, status, signs_text, segmenter


_examples = [
    str(ROOT / "test" / name)
    for name in ("cam-on.mp4", "anh.mp4", "em.mp4")
    if (ROOT / "test" / name).exists()
]

with gr.Blocks(title="SignSpeak — Demo dịch ngôn ngữ ký hiệu", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤟 SignSpeak — Demo dịch ngôn ngữ ký hiệu tiếng Việt\n"
        "Ba chế độ: **Video** (quay/tải lên; hỗ trợ 1 hoặc nhiều từ), "
        "**Realtime** (webcam liên tục, tự cắt + tự đọc khi nghỉ tay), và "
        "**Từng từ** (webcam: ký một từ rồi hạ tay để chốt, lặp lại).\n"
        "*Mẹo: ngồi lùi thấy nửa người trên + hai tay, nền gọn, đủ sáng.*"
    )

    with gr.Tabs():
        with gr.Tab("🎬 Video từng ký hiệu"):
            signs_state = gr.State([])
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(
                        sources=["upload", "webcam"],
                        label="Video ký hiệu (~2-3 giây)",
                    )
                    multi_checkbox = gr.Checkbox(
                        value=False,
                        label="Video có nhiều từ (tự động tách theo cử động tay — nhớ hạ tay giữa các từ)",
                    )
                    recognize_btn = gr.Button("🔍 Nhận dạng ký hiệu", variant="primary")
                    gloss_out = gr.Label(num_top_classes=5, label="Kết quả nhận dạng (top-5)")
                    if _examples:
                        gr.Examples(examples=_examples, inputs=video_in, label="Ví dụ mẫu")

                with gr.Column():
                    signs_box = gr.Textbox(
                        label="Các từ đã nhận dạng (theo thứ tự)", interactive=False, lines=2
                    )
                    with gr.Row():
                        speak_btn = gr.Button("🔊 Tạo câu & đọc", variant="primary")
                        clear_btn = gr.Button("🗑️ Xóa")
                    sentence_out = gr.Textbox(label="Câu tiếng Việt (Qwen3)", lines=2)
                    audio_out = gr.Audio(label="Giọng nói (VieNeu-TTS)", autoplay=True)

            recognize_btn.click(
                recognize,
                inputs=[video_in, signs_state, multi_checkbox],
                outputs=[gloss_out, signs_state, signs_box],
            )
            speak_btn.click(
                make_sentence_and_speak,
                inputs=[signs_state],
                outputs=[sentence_out, audio_out],
            )
            clear_btn.click(
                clear_all,
                inputs=None,
                outputs=[signs_state, signs_box, sentence_out, audio_out, gloss_out],
            )

        with gr.Tab("⚡ Realtime"):
            gr.Markdown(
                "Bật webcam rồi **ký hiệu liên tục**: giơ tay lên để bắt đầu, hạ tay xuống "
                "để kết thúc — hệ thống tự cắt từng ký hiệu và nhận dạng ngay. Ký xong hết "
                "thì bấm **Tạo câu & đọc**."
            )
            rt_signs_state = gr.State([])
            rt_seg_state = gr.State(None)
            with gr.Row():
                with gr.Column():
                    rt_cam = gr.Image(
                        sources=["webcam"], streaming=True, type="numpy", label="Webcam"
                    )
                    rt_status = gr.Markdown("🟢 Sẵn sàng — hãy bắt đầu ký hiệu.")
                    with gr.Accordion("⚙️ Tinh chỉnh phát hiện ký hiệu", open=False):
                        rt_angle = gr.Slider(
                            90, 175, value=150, step=5,
                            label="Ngưỡng góc khuỷu (nhỏ hơn = tay phải giơ cao/gập hơn mới tính là 'đang ký')",
                        )
                        rt_min_down = gr.Slider(
                            2, 20, value=8, step=1,
                            label="Số frame tay hạ liên tục để chốt kết thúc ký hiệu",
                        )
                        rt_idle = gr.Slider(
                            1.0, 8.0, value=3.0, step=0.5,
                            label="Giây nghỉ tay để tự động ghép câu + đọc",
                        )
                with gr.Column():
                    rt_signs_box = gr.Textbox(
                        label="Các từ đã nhận dạng (theo thứ tự)", interactive=False, lines=2
                    )
                    with gr.Row():
                        rt_speak_btn = gr.Button("🔊 Tạo câu & đọc", variant="primary")
                        rt_clear_btn = gr.Button("🗑️ Xóa")
                    rt_sentence_out = gr.Textbox(label="Câu tiếng Việt (Qwen3)", lines=2)
                    rt_audio_out = gr.Audio(label="Giọng nói (VieNeu-TTS)", autoplay=True)

            rt_cam.stream(
                realtime_step,
                inputs=[rt_cam, rt_signs_state, rt_seg_state, rt_angle, rt_min_down, rt_idle],
                outputs=[rt_signs_state, rt_status, rt_signs_box, rt_seg_state, rt_sentence_out, rt_audio_out],
                stream_every=0.1,
                concurrency_limit=1,
                show_progress="hidden",
            )
            rt_speak_btn.click(
                make_sentence_and_speak,
                inputs=[rt_signs_state],
                outputs=[rt_sentence_out, rt_audio_out],
            )
            rt_clear_btn.click(
                realtime_reset,
                inputs=None,
                outputs=[rt_signs_state, rt_status, rt_signs_box, rt_sentence_out, rt_audio_out, rt_seg_state],
            )

        with gr.Tab("✋ Từng từ (hạ tay)"):
            gr.Markdown(
                "Bật webcam. Với **mỗi từ**: giơ tay lên ký → **hạ tay xuống** để chốt từ đó "
                "→ ký từ tiếp theo. Mỗi lần hạ tay là một từ được nhận dạng. Ký xong tất cả "
                "thì bấm **Tạo câu & đọc** (chế độ này KHÔNG tự đọc)."
            )
            pw_signs_state = gr.State([])
            pw_seg_state = gr.State(None)
            with gr.Row():
                with gr.Column():
                    pw_cam = gr.Image(
                        sources=["webcam"], streaming=True, type="numpy", label="Webcam"
                    )
                    pw_status = gr.Markdown("🟢 Sẵn sàng — ký một từ rồi hạ tay.")
                    with gr.Accordion("⚙️ Tinh chỉnh phát hiện ký hiệu", open=False):
                        pw_angle = gr.Slider(
                            90, 175, value=150, step=5,
                            label="Ngưỡng góc khuỷu (nhỏ hơn = tay phải giơ cao/gập hơn mới tính là 'đang ký')",
                        )
                        pw_min_down = gr.Slider(
                            2, 20, value=6, step=1,
                            label="Số frame tay hạ liên tục để chốt một từ",
                        )
                with gr.Column():
                    pw_signs_box = gr.Textbox(
                        label="Các từ đã nhận dạng (theo thứ tự)", interactive=False, lines=2
                    )
                    with gr.Row():
                        pw_speak_btn = gr.Button("🔊 Tạo câu & đọc", variant="primary")
                        pw_clear_btn = gr.Button("🗑️ Xóa")
                    pw_sentence_out = gr.Textbox(label="Câu tiếng Việt (Qwen3)", lines=2)
                    pw_audio_out = gr.Audio(label="Giọng nói (VieNeu-TTS)", autoplay=True)

            pw_cam.stream(
                perword_step,
                inputs=[pw_cam, pw_signs_state, pw_seg_state, pw_angle, pw_min_down],
                outputs=[pw_signs_state, pw_status, pw_signs_box, pw_seg_state],
                stream_every=0.1,
                concurrency_limit=1,
                show_progress="hidden",
            )
            pw_speak_btn.click(
                make_sentence_and_speak,
                inputs=[pw_signs_state],
                outputs=[pw_sentence_out, pw_audio_out],
            )
            pw_clear_btn.click(
                realtime_reset,
                inputs=None,
                outputs=[pw_signs_state, pw_status, pw_signs_box, pw_sentence_out, pw_audio_out, pw_seg_state],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
