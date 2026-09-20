"""WS /ws/keypoints -- luong chinh: nhan keypoint stream, emit gloss + cau + WAV.

Moi ket noi la mot phien doc lap. Luong nay SEGMENT theo ranh gioi (KeypointSegmenter):
tay gio len = bat dau ky hieu, tay ha xuong = ket thuc -> cat clip tron 1 ky hieu (dai bien
thien) roi late-fusion SPOTER + SL-GCN. Nguoi ky HA TAY giua cac tu de cat sach.
"""

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from serving.config import settings
from serving.dependencies import get_gloss_decoder, get_llm, get_spoter, get_tts
from serving.models.schemas import KeypointFrame
from serving.models.session import SessionState
from serving.services.segmenter import KeypointSegmenter
from serving.utils.logger import get_logger
from serving.utils.warmup import ensure_tts_loaded

router = APIRouter()
logger = get_logger(__name__)


def _make_segmenter() -> KeypointSegmenter:
    return KeypointSegmenter(
        angle_threshold=settings.seg_angle_threshold,
        min_up_frames=settings.seg_min_up_frames,
        min_down_frames=settings.seg_min_down_frames,
        min_sign_frames=settings.seg_min_sign_frames,
        max_sign_frames=settings.seg_max_sign_frames,
        preroll_frames=settings.seg_preroll_frames,
        min_wrist_motion_px=settings.seg_min_wrist_motion_px,
        min_hand_frames_ratio=settings.min_hand_frames_ratio,
    )


async def _finalize(ws: WebSocket, session: SessionState) -> None:
    """Ghep cau (LLM) + doc (TTS) khi client bao end_sign.

    LUON ket thuc bang mot ban tin "done": client cho tin nhac nay de thoat trang thai
    "dang xu ly". Thieu no thi mot buoc loi (LLM/TTS) lam client treo den khi het timeout.
    """
    await ws.send_json({"type": "processing"})

    llm = get_llm()
    try:
        sentence = await llm.gloss_to_sentence(session.gloss_seq)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM loi (%r); ghep tu tho.", exc)
        sentence = " ".join(session.gloss_seq)
    # LLM tra ve rong (bi cat sach boi _strip_think chang han) van phai co cau de doc.
    if not sentence.strip():
        sentence = " ".join(session.gloss_seq)
    await ws.send_json({"type": "sentence", "text": sentence})

    has_audio = False
    try:
        await ensure_tts_loaded()
        wav_bytes = await get_tts().synthesize(sentence)
        await ws.send_bytes(wav_bytes)
        has_audio = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS loi (%r); bo qua giong noi.", exc)

    await ws.send_json({"type": "done", "has_audio": has_audio})
    session.reset()


@router.websocket("/ws/keypoints")
async def keypoint_stream(ws: WebSocket):
    await ws.accept()
    spoter = get_spoter()
    decoder = get_gloss_decoder()
    session = SessionState(_make_segmenter())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = KeypointFrame.model_validate_json(raw)
            except ValidationError as exc:
                await ws.send_json({"type": "error", "error": str(exc)})
                continue

            frame = np.array(data.keypoints, dtype=np.float32)
            slgcn_frame = (
                np.array(data.slgcn_keypoints, dtype=np.float32)
                if data.slgcn_keypoints is not None
                else None
            )
            # Segment theo tay len/xuong: chi tra ve clip khi vua KET THUC mot ky hieu.
            clip = session.segmenter.push(frame, slgcn_frame)
            if clip is not None:
                await _classify_and_emit(ws, session, spoter, decoder, clip)

            if data.end_sign:
                # Chot not ky hieu dang do (neu co) truoc khi ghep cau.
                pending = session.segmenter.flush()
                if pending is not None:
                    await _classify_and_emit(ws, session, spoter, decoder, pending)
                if session.gloss_seq:
                    await _finalize(ws, session)
                else:
                    # Khong ky hieu nao du tin cay: van phai bao client de no thoat trang
                    # thai cho (truoc day im lang -> LLM/TTS coi nhu dung theo).
                    logger.info("Ket thuc phien nhung khong co gloss nao du tin cay.")
                    await ws.send_json({"type": "no_sign"})
                    session.reset()

    except WebSocketDisconnect:
        logger.info("Client ngat ket noi.")


async def _classify_and_emit(ws, session, spoter, decoder, clip) -> None:
    """Late-fusion mot clip 1-ky-hieu -> emit gloss neu du tin cay."""
    spoter_clip, slgcn_clip = clip
    # LATE-FUSION SPOTER + SL-GCN (tu lui ve SPOTER-only neu thieu slgcn_clip).
    idx, conf, fused = await spoter.fused_infer(spoter_clip, slgcn_clip)
    # Fusion lam phan phoi phang hon -> dung nguong rieng cho realtime; SPOTER-only giu 0.60.
    threshold = (
        settings.realtime_fusion_confidence_threshold
        if fused
        else settings.spoter_confidence_threshold
    )
    if conf < threshold:
        logger.debug("Bo gloss do tin cay thap (%.3f < %.3f).", conf, threshold)
        return
    # Moi clip la mot ky hieu rieng biet (da tach boi ha tay) -> them thang.
    gloss = decoder.label(idx)
    session.gloss_seq.append(gloss)
    await ws.send_json({"type": "gloss", "gloss": gloss, "confidence": conf})
