"""Pre-process demo videos into the result cache (keyed by file sha256) for instant replay."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

# Allow running directly via `python scripts/prebake_videos.py` (root = backend/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from serving.config import settings  # noqa: E402
from serving.dependencies import (  # noqa: E402
    get_gloss_decoder,
    get_llm,
    get_result_cache,
    get_slgcn,
    get_spoter,
    get_tts,
    get_video_recognizer,
)
from serving.utils.logger import get_logger  # noqa: E402

logger = get_logger("prebake")

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def sha256_file(path: Path) -> str:
    """Same algorithm as the client (crypto.subtle SHA-256 of file contents)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_duration(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return count / fps if fps > 0 and count > 0 else None


def spread_glosses(glosses: list[str], duration: float | None) -> list[dict]:
    """Spread words evenly over duration; video-file path has no real per-segment
    timestamps (approximate until overwritten by a live run)."""
    if not glosses:
        return []
    if not duration or duration <= 0:
        return [{"text": g, "at": 0.0} for g in glosses]
    step = duration / (len(glosses) + 1)
    return [{"text": g, "at": round(step * (i + 1), 3)} for i, g in enumerate(glosses)]


def collect_videos(inputs: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES))
        elif path.is_file():
            found.append(path)
        else:
            logger.warning("Bo qua %s (khong ton tai).", path)
    return found


def list_cache() -> None:
    cache_dir = settings.result_cache_dir
    entries = sorted(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    if not entries:
        print(f"Cache trong ({cache_dir}).")
        return
    print(f"{len(entries)} video da xu ly ({cache_dir}):")
    for path in entries:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"  {path.stem}  <doc khong duoc>")
            continue
        has_wav = path.with_suffix(".wav").is_file()
        words = ", ".join(g["text"] for g in entry.get("glosses", []))
        print(
            f"  {path.stem[:16]}...  [{entry.get('source', '?')}]"
            f"{' +wav' if has_wav else '      '}  {words}  ->  {entry.get('sentence', '')}"
        )


async def set_sentence(path: Path, sentence: str, with_tts: bool) -> bool:
    """Overwrite the sentence for one video (keeps recognized words/timestamps) and
    re-synthesizes speech; use when the LLM mis-joined a long clip."""
    cache = get_result_cache()
    key = sha256_file(path)
    entry = cache.get(key)
    if entry is None:
        logger.info("%s: chua co trong cache -> nhan dang truoc da.", path.name)
        await prebake_one(path, force=False, with_tts=False)
        entry = cache.get(key)
        if entry is None:
            logger.error("%s: khong nhan dang duoc -> khong ghi duoc cau.", path.name)
            return False

    wav_bytes = None
    if with_tts:
        try:
            wav_bytes = await get_tts().synthesize(sentence)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: TTS loi (%r) -> luu cau khong kem giong noi.", path.name, exc)

    cache.put(
        key,
        glosses=entry["glosses"],
        sentence=sentence,
        duration=entry.get("duration"),
        wav_bytes=wav_bytes,
        source=entry.get("source", "prebake"),
    )
    logger.info("%s: da ghi cau -> %s", path.name, sentence)
    return True


async def prebake_one(path: Path, force: bool, with_tts: bool) -> bool:
    cache = get_result_cache()
    key = sha256_file(path)
    if not force and cache.get(key) is not None:
        logger.info("%s: da co trong cache -> bo qua (dung --force de chay lai).", path.name)
        return False

    logger.info("%s: dang nhan dang...", path.name)
    glosses = await get_video_recognizer().recognize(str(path))
    if not glosses:
        logger.warning("%s: khong nhan duoc tu nao -> khong luu cache.", path.name)
        return False

    try:
        sentence = await get_llm().gloss_to_sentence(glosses)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: LLM loi (%r) -> ghep tu tho.", path.name, exc)
        sentence = " ".join(glosses)
    if not sentence.strip():
        sentence = " ".join(glosses)

    wav_bytes = None
    if with_tts:
        try:
            wav_bytes = await get_tts().synthesize(sentence)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: TTS loi (%r) -> luu cache khong kem giong noi.", path.name, exc)

    duration = video_duration(path)
    cache.put(
        key,
        glosses=spread_glosses(glosses, duration),
        sentence=sentence,
        duration=duration,
        wav_bytes=wav_bytes,
        source="prebake",
    )
    logger.info("%s: %s -> %s", path.name, " / ".join(glosses), sentence)
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="File video hoac thu muc chua video")
    parser.add_argument("--force", action="store_true", help="Chay lai ca video da co trong cache")
    parser.add_argument("--no-tts", action="store_true", help="Bo buoc tao giong noi")
    parser.add_argument("--list", action="store_true", help="Chi liet ke cache hien co")
    parser.add_argument(
        "--sentence",
        help="Ghi de CAU cho DUNG MOT video (giu nguyen tu da nhan dang) va tao lai giong noi",
    )
    args = parser.parse_args()

    if args.list:
        list_cache()
        return 0
    videos = collect_videos(args.inputs)
    if not videos:
        parser.error("Khong co video nao de xu ly.")
    if args.sentence and len(videos) != 1:
        parser.error("--sentence chi ap dung cho DUNG MOT video.")

    # Load only what is needed (skip warmup_all to avoid loading unused models).
    get_gloss_decoder().load()
    get_spoter().load()
    try:
        get_slgcn().load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nap SL-GCN that bai (%r) -> chay SPOTER-only, do chinh xac thap hon.", exc)
    get_video_recognizer().load()
    get_llm().load()
    if not args.no_tts:
        get_tts().load()

    done = 0
    for path in videos:
        try:
            if args.sentence:
                done += await set_sentence(path, args.sentence, not args.no_tts)
                continue
            done += await prebake_one(path, args.force, not args.no_tts)
        except Exception as exc:  # noqa: BLE001
            logger.error("%s: loi (%r) -> bo qua.", path.name, exc)
    await get_llm().close()
    logger.info("Xong: %d/%d video moi duoc luu vao %s", done, len(videos), settings.result_cache_dir)
    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    # MediaPipe/TFLite can segfault on interpreter exit after work is done; os._exit
    # avoids that so the exit code reflects the real result.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
