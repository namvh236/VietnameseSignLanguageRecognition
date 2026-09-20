"""Cau hinh tap trung cho backend (pydantic-settings).

Moi hang so, duong dan deu doc o day. AI core (src/), artifact model (ONNX, gloss.csv)
nam CUNG mot cay voi serving/ (goc du an = backend/), nen recognition_root mac dinh la
chinh goc du an; property tra ve duong dan tuyet doi.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# serving/config.py -> parent.parent = goc du an (backend/), noi chua src/, models/, experiments/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Goc du an (chua src/pipelines, models/*.onnx, experiments/gloss.csv). Ghi de qua env neu can.
    recognition_root: Path = _PROJECT_ROOT

    # SPOTER (ONNX). File spoter_v3.onnx la ban re-export self-contained, XUAT SAN
    # "probabilities" (softmax trong graph) -> KHONG softmax lai trong code (xem
    # docs/local_fusion_deployment.md). Thay cho cap cu spoter_multicam_v3.onnx (+ .onnx.data).
    spoter_onnx_name: str = "models/spoter/spoter_v3.onnx"
    gloss_csv_name: str = "experiments/gloss.csv"
    spoter_num_classes: int = 400
    spoter_seq_len: int = 70
    spoter_confidence_threshold: float = 0.60
    spoter_providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # SL-GCN ensemble (ONNX) — nhanh thu 2 cua late-fusion. Cung xuat san "probabilities"
    # (bone/motion/normalize/softmax-avg 4 stream deu bake trong graph). CHI dung cho luong
    # video-file (co Pose day du); luong realtime/REST keypoint van SPOTER-only vi client chi
    # gui 54 khop SPOTER (khong co confidence / khong du khop cho SL-GCN).
    slgcn_onnx_name: str = "models/sl-gcn/sl_gcn_ensemble.onnx"
    slgcn_seq_len: int = 150   # cua so SL-GCN (khac SPOTER = 70)
    slgcn_num_points: int = 27  # so joint SL-GCN
    # Fusion: P = fusion_w_slgcn * P_slgcn + (1 - fusion_w_slgcn) * P_spoter. Trong so da chot
    # tren validation/test (SL-GCN 0.75 / SPOTER 0.25 -> fusion 90.1%).
    fusion_w_slgcn: float = 0.75
    # Nguong tin cay cho luong REALTIME/REST khi fusion (phan phoi fusion phang hon SPOTER-only,
    # xem video_confidence_threshold). Khi client KHONG gui keypoint SL-GCN -> lui ve SPOTER-only
    # va dung spoter_confidence_threshold. Se can tinh chinh tren validation.
    fusion_confidence_threshold: float = 0.12
    # Rieng luong WS REALTIME dung nguong CAO hon (0.12 -> 0.30): khac luong video-file (nguoi
    # dung chu dong bam nhan dang mot clip da quay san), realtime chay lien tuc nen moi doan rac
    # deu co co hoi thanh gloss ao. Trong so fusion KHONG doi (van 0.75/0.25).
    realtime_fusion_confidence_threshold: float = 0.30

    # Loc cua so "co dang ky hieu" truoc khi goi model (chong false positive khi
    # tay nghi hoac khong xuat hien trong camera — model luon tra ve MOT lop nao do).
    min_hand_frames_ratio: float = 0.3  # ti le frame trong cua so co it nhat 1 ban tay
    min_hand_motion_px: float = 12.0  # bien do di chuyen tay toi thieu trong cua so (pixel)

    # Ollama / Qwen3
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout: float = 10.0
    # Toi uu toc do infer:
    # - keep_alive: giu model trong VRAM giua cac request, tranh nap lai (cold start).
    # - num_predict: cau ghep rat ngan, gioi han token sinh ra de tra ket qua nhanh.
    ollama_keep_alive: str = "30m"
    ollama_num_predict: int = 128

    # TTS (VieNeu). Sample rate that lay tu engine luc chay; day chi la du phong.
    tts_mode: str = "standard"
    tts_sample_rate: int = 24000

    # Session buffer (con dung cho REST fallback; luong WS realtime da chuyen sang segment
    # theo ranh gioi — xem duoi).
    session_stride: int = 10

    # Segment theo ranh gioi cho luong WS realtime (thay cua so truot co dinh).
    # Tinh goc canh tay tu chinh keypoint dang stream: tay gio len = bat dau ky hieu,
    # tay ha xuong = ket thuc -> cat clip tron 1 ky hieu (dai bien thien), moi nhanh pad
    # theo contract rieng (SPOTER 70 / SL-GCN 150). Port tu RealtimeSegmenter cua demo.
    # Nguoi ky HA TAY giua cac tu de cat sach (model la isolated-sign, khong phai CSLR).
    # Goc khuyu < nguong = "tay dang gio" (do). 150 qua CAO: ngoi truoc webcam, tay de tren ban
    # /tren dui thi khuyu da gap ~90-140 do -> luon bi coi la dang ky hieu (tay buong thang moi
    # ~180). 115 doi hoi tay thuc su gio len. Segmenter con doi CO BAN TAY duoc phat hien nua.
    seg_angle_threshold: float = 115.0
    seg_min_up_frames: int = 3           # so frame tay-len lien tuc de bat dau
    # So frame tay-xuong lien tuc de chot ket thuc. DE CAO (12) vi keypoint stream KHONG
    # duoc smooth (demo dung Pose smooth_landmarks=True): mot khoang khung nhe GIUA ky hieu
    # (~10 frame khi tay hoi duoi thang) khong duoc coi la ket thuc; ha tay CO Y giua cac tu
    # dai hon han. Neu van bi cat giua ky hieu -> tang them.
    seg_min_down_frames: int = 12        # so frame tay-xuong lien tuc de chot ket thuc
    seg_min_sign_frames: int = 8         # clip ngan hon -> bo (nhieu)
    # Cap an toan: clip dai hon -> DUNG lai va BO clip (khong doan nhan). Tin hieu cat chinh la
    # HA TAY; clip cham cap tuc la chua he thay tay ha -> gan nhu chac chan la doan rac (tay cu
    # trong khung). Mot ky hieu don co the dai ~4-5s (vd "Em" ~136 frame @30fps) nen giu 160.
    seg_max_sign_frames: int = 160
    # Dem truoc (frame giu lai TRUOC khi phat hien tay-len). DE CAO (20): SPOTER Pad(70) chi
    # giu 70 frame dau, va model duoc train tren clip CO doan dan nhap (tay dang gio len). Cat
    # sat qua (preroll nho) lam mat doan do -> sai (vd em->"Sớm", nam-yeu-thuong->"Cái quần").
    # preroll=20 phuc hoi ca hai (test len 8/8). Trong luong lien tuc, day chinh la doan nghi/
    # gio tay giua 2 tu (nguoi ky ha tay ro giua cac tu) -> khong lan sang tu truoc.
    seg_preroll_frames: int = 20         # dem truoc de khong cat cut dau ky hieu
    # Bien do di chuyen co tay toi thieu (pixel) -> loc tu the nghi. 12 qua THAP: jitter cua
    # MediaPipe tren mot clip vai giay da vuot qua. Ky hieu that di chuyen co tay hang tram pixel.
    seg_min_wrist_motion_px: float = 40.0

    # Nhan dang tu file video (upload/quay) — dung LATE-FUSION.
    # LUU Y QUAN TRONG: xac suat FUSION phang hon SPOTER-only rat nhieu (SL-GCN 0.75 la
    # softmax-avg 4 stream, top-1 thuong chi ~0.1-0.4), nen nguong nay THAP hon han cai cu
    # (0.5, von hop cho logits->softmax cua SPOTER-only). Day chi la nguong loc doan rac
    # (transition/tu the nghi) — segment_video da loc chinh bang goc canh tay + cu dong tay.
    # NEN tinh chinh lai tren tap validation cho phan phoi fusion; 0.12 la mac dinh tam.
    video_confidence_threshold: float = 0.12
    max_upload_mb: int = 100  # gioi han kich thuoc moi file upload (khop UI frontend)
    # Toi uu toc do infer video: MediaPipe la nut co chai (~28 fps/luong o do phuc tap 1).
    # - Pose (phan doan) = 0: chi can goc canh tay tho -> nhanh hon, KHONG anh huong do chinh
    #   xac phan loai (phan loai chay Holistic rieng). Do la mac dinh.
    # - Holistic (phan loai) = 1: do 0 lam TUT do chinh xac (thu nghiem: mat ky hieu "Em"),
    #   nen giu 1. Muon nhanh hon (chap nhan giam chinh xac) co the dat 0 qua env.
    video_pose_model_complexity: int = 0  # MediaPipe Pose cho phan doan (0/1/2)
    video_holistic_model_complexity: int = 1  # MediaPipe Holistic cho phan loai (0/1/2)

    # Cache ket qua theo file video (an toan cho demo): video da xu ly mot lan thi lan sau
    # phat lai ngay, khong chay lai MediaPipe/ONNX/LLM/TTS. Xem services/result_cache.py.
    result_cache_enabled: bool = True
    result_cache_dir_name: str = "cache/results"

    # Server
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
