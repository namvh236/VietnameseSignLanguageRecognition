"""Request/response schemas. Keypoint contract: keypoints = 108 floats (54 joints, x,y
pixel) in JointSelect order: 12 body joints, then 21 left-hand, then 21 right-hand.
Joint 1 (neck) must be [0, 0] -- the model was trained with neck zeroed out."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

NUM_JOINTS = 54
KEYPOINTS_LEN = NUM_JOINTS * 2
SLGCN_NUM_JOINTS = 27
SLGCN_KEYPOINTS_LEN = SLGCN_NUM_JOINTS * 3  # 81
# TTS text length cap (guards huge requests; gloss-joined sentences are far shorter).
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
    """Fallback one-shot REST: list of keypoint frames for ONE sentence. slgcn_frames
    (optional) runs parallel to frames; omitting it falls back to SPOTER-only.
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
    """Recognition result for ONE video: ordered list of recognized words (glosses)."""

    glosses: list[str]


class ComposeRequest(BaseModel):
    """Joins a word list into a sentence + voice."""

    glosses: list[str] = Field(..., min_length=1)


class ComposeResponse(BaseModel):
    sentence: str
    # WAV base64 (frontend decodes via base64ToAudioUrl); None if TTS is unavailable.
    audio_b64: str | None = None
    mime: str = "audio/wav"


class CachedGloss(BaseModel):
    """One cached word, with its timestamp in the video (seconds)."""

    text: str
    at: float = 0.0


class CachedResultPayload(BaseModel):
    """Sent by the client after a successful run, so it isn't rerun next time."""

    glosses: list[CachedGloss] = Field(..., min_length=1)
    sentence: str
    duration: float | None = None
    # WAV base64 from that run (client already has it); None -> cache has no voice.
    audio_b64: str | None = None


class CachedResultResponse(BaseModel):
    glosses: list[CachedGloss]
    sentence: str
    duration: float | None = None
    audio_b64: str | None = None
    mime: str = "audio/wav"
    # 'live' = result from an actual run on the web; 'prebake' = pre-run by a script.
    source: str = "live"
