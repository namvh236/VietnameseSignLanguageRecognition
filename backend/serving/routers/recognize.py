"""POST /api/recognize -- fallback one-shot: keypoint frame list -> gloss + sentence."""

import numpy as np
from fastapi import APIRouter

from serving.config import settings
from serving.dependencies import get_gloss_decoder, get_llm, get_spoter
from serving.models.schemas import RecognizeRequest, RecognizeResponse
from serving.services.frame_buffer import FrameBuffer
from serving.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest):
    spoter = get_spoter()
    decoder = get_gloss_decoder()
    buffer = FrameBuffer(settings.spoter_seq_len, settings.session_stride)
    glosses: list[str] = []

    # slgcn_frames (optional) runs parallel for late-fusion; SPOTER-only if absent.
    slgcn_frames = req.slgcn_frames
    for i, frame in enumerate(req.frames):
        slgcn_frame = (
            np.array(slgcn_frames[i], dtype=np.float32)
            if slgcn_frames is not None and i < len(slgcn_frames)
            else None
        )
        result = buffer.push(np.array(frame, dtype=np.float32), slgcn_frame)
        if result is None:
            continue
        snapshot, slgcn_snapshot = result
        idx, conf, fused = await spoter.fused_infer(snapshot, slgcn_snapshot)
        threshold = (
            settings.fusion_confidence_threshold
            if fused
            else settings.spoter_confidence_threshold
        )
        gloss = decoder.label(idx) if conf >= threshold else None
        if gloss:
            glosses.append(gloss)

    if not glosses:
        return RecognizeResponse(glosses=[], sentence="")

    try:
        sentence = await get_llm().gloss_to_sentence(glosses)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM loi (%r); ghep tu tho.", exc)
        sentence = " ".join(glosses)

    return RecognizeResponse(glosses=glosses, sentence=sentence)
