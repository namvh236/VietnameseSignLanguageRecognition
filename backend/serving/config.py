"""Centralized backend settings; recognition_root defaults to the project root."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# serving/config.py -> parent.parent = project root (backend/), with src/, models/, etc.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    recognition_root: Path = _PROJECT_ROOT

    # spoter_v3.onnx is self-contained and already applies softmax in-graph
    # ("probabilities" output) -> do not softmax again.
    spoter_onnx_name: str = "models/spoter/spoter_v3.onnx"
    gloss_csv_name: str = "experiments/gloss.csv"
    spoter_num_classes: int = 400
    spoter_seq_len: int = 70
    spoter_confidence_threshold: float = 0.60
    spoter_providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # SL-GCN ensemble (ONNX): 2nd late-fusion branch, also bakes in "probabilities".
    # Video-file flow only; realtime/REST stays SPOTER-only since clients send just the
    # 54 SPOTER joints (no confidence, not enough joints for SL-GCN).
    slgcn_onnx_name: str = "models/sl-gcn/sl_gcn_ensemble.onnx"
    slgcn_seq_len: int = 150  # SL-GCN window length (SPOTER's is 70)
    slgcn_num_points: int = 27
    # P = fusion_w_slgcn * P_slgcn + (1 - fusion_w_slgcn) * P_spoter.
    # Weight tuned on validation/test (0.75/0.25 -> 90.1% fusion accuracy).
    fusion_w_slgcn: float = 0.75
    # Confidence threshold for realtime/REST fusion. Fused distribution is flatter than
    # SPOTER-only, hence lower than spoter_confidence_threshold; falls back to that
    # threshold when the client sends no SL-GCN keypoints.
    fusion_confidence_threshold: float = 0.12
    # Realtime WS needs a higher threshold (0.30 vs 0.12): unlike one-shot video clips,
    # continuous streaming gives every noisy segment a chance to become a false gloss.
    # Fusion weight is unchanged (0.75/0.25).
    realtime_fusion_confidence_threshold: float = 0.30

    # Filters out windows that don't "look like signing" before calling the model
    # (it always returns some class, even for idle/off-camera hands).
    min_hand_frames_ratio: float = 0.3  # fraction of frames with >=1 hand visible
    min_hand_motion_px: float = 12.0  # min hand motion within the window (px)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout: float = 10.0
    # Inference speed: keep_alive keeps the model in VRAM between requests (avoids cold
    # start); num_predict caps generated tokens since output sentences are short.
    ollama_keep_alive: str = "30m"
    ollama_num_predict: int = 128

    # tts_sample_rate is only a fallback; real rate comes from the engine at runtime.
    tts_mode: str = "standard"
    tts_sample_rate: int = 24000

    # Still used by the REST fallback; realtime WS has moved to boundary-based
    # segmenting (below).
    session_stride: int = 10

    # Boundary-based segmentation for realtime WS (replaces a fixed sliding window):
    # elbow angle marks hand-up (start) / hand-down (end). Ported from the demo's
    # RealtimeSegmenter; isolated-sign model needs hands dropped between words.

    # Elbow angle below this = "hand up". 115 is deliberately strict: resting arms on a
    # desk/lap bend the elbow ~90-140 too, which a looser threshold misreads as signing.
    seg_angle_threshold: float = 115.0
    seg_min_up_frames: int = 3  # consecutive up-frames to start a sign
    # High (12 vs. typical ~3-5): the keypoint stream isn't smoothed, so a brief
    # mid-sign dip (~10 frames) must not be read as the end of the sign.
    seg_min_down_frames: int = 12
    seg_min_sign_frames: int = 8  # clips shorter than this are dropped as noise
    # Safety cap: no hand-down by this length is almost certainly a stale frame, not a
    # genuine sign; 160 leaves headroom over the longest observed sign (~136 frames).
    seg_max_sign_frames: int = 160
    # High (20): SPOTER's Pad(70) keeps only the first 70 frames, and the model is
    # trained on clips with a lead-in (hand already rising). A smaller preroll cuts
    # that lead-in and misclassifies signs (validated at 8/8 with 20).
    seg_preroll_frames: int = 20
    # 12 was too low: MediaPipe jitter over a multi-second clip exceeds that; real signs
    # move the wrist hundreds of pixels.
    seg_min_wrist_motion_px: float = 40.0

    # Late-fusion video recognition. Fused probabilities are much flatter than
    # SPOTER-only (top-1 often ~0.1-0.4), hence far lower than the old SPOTER-only 0.5;
    # segment_video already filters noise via arm angle + wrist motion. Needs retuning.
    video_confidence_threshold: float = 0.12
    max_upload_mb: int = 100  # must match the frontend upload limit
    # MediaPipe is the bottleneck (~28 fps/stream at complexity 1). Pose (segmentation)
    # defaults to 0 since only a rough arm angle is needed; Holistic stays at 1
    # because 0 measurably hurts accuracy (lost sign "Em" in testing).
    video_pose_model_complexity: int = 0
    video_holistic_model_complexity: int = 1

    # Caches recognition results per video file so replaying the same video skips
    # re-running MediaPipe/ONNX/LLM/TTS.
    result_cache_enabled: bool = True
    result_cache_dir_name: str = "cache/results"

    server_host: str = "0.0.0.0"
    server_port: int = 8000
    cors_allow_origins: list[str] = ["*"]

    @property
    def recognition_src(self) -> Path:
        return self.recognition_root / "src"

    @property
    def spoter_onnx_path(self) -> Path:
        return self.recognition_root / self.spoter_onnx_name

    @property
    def slgcn_onnx_path(self) -> Path:
        return self.recognition_root / self.slgcn_onnx_name

    @property
    def gloss_csv_path(self) -> Path:
        return self.recognition_root / self.gloss_csv_name

    @property
    def result_cache_dir(self) -> Path:
        return self.recognition_root / self.result_cache_dir_name


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
