"""WS /ws/keypoints -- main stream: receives keypoints, emits gloss + sentence + WAV.
Each connection is independent, segmented by KeypointSegmenter: hand up starts a sign,
hand down ends it; signer must drop hands between words to cut cleanly."""

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
    """Composes a sentence (LLM) + voice (TTS) when the client reports end_sign. ALWAYS
    ends with a "done" message -- without it, an LLM/TTS error leaves the client hanging
    in "processing" state until timeout."""
    await ws.send_json({"type": "processing"})

    llm = get_llm()
    try:
        sentence = await llm.gloss_to_sentence(session.gloss_seq)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM loi (%r); ghep tu tho.", exc)
        sentence = " ".join(session.gloss_seq)
    # Empty LLM output (e.g. fully stripped by _strip_think) still needs a sentence.
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
            # Segments by hand up/down: returns a clip only when a sign just ended.
            clip = session.segmenter.push(frame, slgcn_frame)
            if clip is not None:
                await _classify_and_emit(ws, session, spoter, decoder, clip)

            if data.end_sign:
                # Flush any in-progress sign before composing the sentence.
                pending = session.segmenter.flush()
                if pending is not None:
                    await _classify_and_emit(ws, session, spoter, decoder, pending)
                if session.gloss_seq:
                    await _finalize(ws, session)
                else:
                    # No sign was confident enough: still tell the client so it leaves
                    # the waiting state (previously silent, which looked like a hang).
                    logger.info("Ket thuc phien nhung khong co gloss nao du tin cay.")
                    await ws.send_json({"type": "no_sign"})
                    session.reset()

    except WebSocketDisconnect:
        logger.info("Client ngat ket noi.")


async def _classify_and_emit(ws, session, spoter, decoder, clip) -> None:
    """Late-fuses one sign clip -> emits a gloss if confident enough."""
    spoter_clip, slgcn_clip = clip
    # Late-fusion SPOTER + SL-GCN (falls back to SPOTER-only if slgcn_clip is missing).
    idx, conf, fused = await spoter.fused_infer(spoter_clip, slgcn_clip)
    # Fusion flattens the distribution; realtime uses its own threshold (0.60 base).
    threshold = (
        settings.realtime_fusion_confidence_threshold
        if fused
        else settings.spoter_confidence_threshold
    )
    if conf < threshold:
        logger.debug("Bo gloss do tin cay thap (%.3f < %.3f).", conf, threshold)
        return
    # Each clip is a distinct sign (already split by hand-down) -> append directly.
    gloss = decoder.label(idx)
    session.gloss_seq.append(gloss)
    await ws.send_json({"type": "gloss", "gloss": gloss, "confidence": conf})
