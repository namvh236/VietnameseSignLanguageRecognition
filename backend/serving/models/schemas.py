"""Schema Pydantic cho request/response.

Hop dong keypoint (QUAN TRONG):
    keypoints = 108 float = 54 khop x (x, y), toa do PIXEL (da nhan voi width/height),
    theo dung thu tu ma JointSelect sinh ra:
        - 12 khop than: nose, neck, rightEye, leftEye, rightEar, leftEar,
          rightShoulder, leftShoulder, rightElbow, leftElbow, rightWrist, leftWrist
        - 21 khop tay TRAI, roi 21 khop tay PHAI (thu tu HAND_LANDMARKS)
    Khop index 1 (neck) PHAI la [0, 0] -- model duoc huan luyen voi neck = 0.

Hop dong keypoint SL-GCN (TUY CHON, cho late-fusion realtime):
    slgcn_keypoints = 81 float = 27 khop x (x, y, confidence), toa do PIXEL, thu tu SLGCN_JOINTS[27]:
        - 7 khop than: nose, leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist
        - 10 khop tay TRAI, roi 10 khop tay PHAI
    confidence: than = visibility cua MediaPipe Pose; tay = 1.0 neu ban tay duoc phat hien, nguoc lai 0.
    Khop thieu = [0, 0, 0]. Neu khong gui -> server tu lui ve SPOTER-only.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

NUM_JOINTS = 54
KEYPOINTS_LEN = NUM_JOINTS * 2
SLGCN_NUM_JOINTS = 27
SLGCN_KEYPOINTS_LEN = SLGCN_NUM_JOINTS * 3  # 81
# Gioi han do dai van ban TTS (chong request qua kho; cau ghep tu gloss ngan hon nhieu).
MAX_TTS_TEXT_LEN = 3200


class KeypointFrame(BaseModel):
    keypoints: list[float] = Field(..., description=f"{KEYPOINTS_LEN} float: 54 khop x (x, y)")
    slgcn_keypoints: list[float] | None = Field(
        default=None, description=f"(tuy chon) {SLGCN_KEYPOINTS_LEN} float: 27 khop x (x, y, confidence)"
    )
    end_sign: bool = False

    @field_validator("keypoints")
    @classmethod
    def _check_len(cls, value: list[float]) -> list[float]:
        if len(value) != KEYPOINTS_LEN:
            raise ValueError(f"Can dung {KEYPOINTS_LEN} gia tri keypoint, nhan duoc {len(value)}")
        return value

    @field_validator("slgcn_keypoints")
    @classmethod
    def _check_slgcn_len(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != SLGCN_KEYPOINTS_LEN:
            raise ValueError(
                f"slgcn_keypoints can dung {SLGCN_KEYPOINTS_LEN} gia tri, nhan duoc {len(value)}"
            )
        return value


class WSMessage(BaseModel):
    type: Literal["gloss", "processing", "sentence", "error"]
    gloss: str | None = None
    confidence: float | None = None
    text: str | None = None
    error: str | None = None


class RecognizeRequest(BaseModel):
    """Fallback one-shot REST: danh sach frame keypoint cho MOT cau.

    slgcn_frames (tuy chon): song song voi frames, moi frame SLGCN_KEYPOINTS_LEN gia tri.
    Neu thieu -> SPOTER-only.
    """

    frames: list[list[float]] = Field(..., min_length=1)
    slgcn_frames: list[list[float]] | None = None

    @field_validator("frames")
    @classmethod
    def _check_shape(cls, value: list[list[float]]) -> list[list[float]]:
        for frame in value:
            if len(frame) != KEYPOINTS_LEN:
                raise ValueError(f"Moi frame can {KEYPOINTS_LEN} gia tri, nhan duoc {len(frame)}")
        return value

    @field_validator("slgcn_frames")
    @classmethod
    def _check_slgcn_shape(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is not None:
            for frame in value:
                if len(frame) != SLGCN_KEYPOINTS_LEN:
                    raise ValueError(
                        f"Moi frame SL-GCN can {SLGCN_KEYPOINTS_LEN} gia tri, nhan duoc {len(frame)}"
                    )
        return value


class RecognizeResponse(BaseModel):
    glosses: list[str]
    sentence: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TTS_TEXT_LEN)


class VideoRecognizeResponse(BaseModel):
    """Ket qua nhan dang MOT video: danh sach tu (gloss) theo thu tu ky hieu."""

    glosses: list[str]


class ComposeRequest(BaseModel):
    """Ghep danh sach tu -> cau + giong noi."""

    glosses: list[str] = Field(..., min_length=1)


class ComposeResponse(BaseModel):
    sentence: str
    # WAV base64 (frontend giai ma bang base64ToAudioUrl); None neu TTS loi/khong san sang.
    audio_b64: str | None = None
    mime: str = "audio/wav"


class CachedGloss(BaseModel):
    """Mot tu trong ket qua da cache, kem moc thoi gian xuat hien trong video (giay)."""

    text: str
    at: float = 0.0


class CachedResultPayload(BaseModel):
    """Client gui len sau mot lan chay THANH CONG de lan sau khong phai chay lai."""

    glosses: list[CachedGloss] = Field(..., min_length=1)
    sentence: str
    duration: float | None = None
    # WAV base64 cua lan chay do (client da co san). None -> cache khong kem giong noi.
    audio_b64: str | None = None


class CachedResultResponse(BaseModel):
    glosses: list[CachedGloss]
    sentence: str
    duration: float | None = None
    audio_b64: str | None = None
    mime: str = "audio/wav"
    # 'live' = ket qua tu mot lan chay that tren web; 'prebake' = script chuan bi truoc.
    source: str = "live"
