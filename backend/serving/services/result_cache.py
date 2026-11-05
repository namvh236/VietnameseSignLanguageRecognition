"""Per-video-file result cache keyed by content hash (sha256): skips re-running
MediaPipe/ONNX/LLM/TTS when a video replays. Keys are validated as hex before touching
the filesystem; layout is cache/results/<key>.{json,wav} (git-untracked)."""

import json
import re
import time
from pathlib import Path

from serving.utils.logger import get_logger

logger = get_logger(__name__)

# Clients prefer sha256 (64 hex); environments without crypto.subtle (http on LAN) fall
# back to a shorter key derived from filename + size + mtime.
KEY_RE = re.compile(r"^[0-9a-f]{16,64}$")


def is_valid_key(key: str) -> bool:
    return bool(KEY_RE.match(key))


class ResultCache:
    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def is_valid_key(key: str) -> bool:
        return is_valid_key(key)

    def _json_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def wav_path(self, key: str) -> Path:
        return self.root / f"{key}.wav"

    def get(self, key: str) -> dict | None:
        """Saved result (with wav_bytes if present), or None if never processed."""
        path = self._json_path(key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache %s doc khong duoc (%r) -> coi nhu chua co.", key, exc)
            return None
        wav = self.wav_path(key)
        entry["wav_bytes"] = wav.read_bytes() if wav.is_file() else None
        return entry

    def put(
        self,
        key: str,
        glosses: list[dict],
        sentence: str,
        duration: float | None = None,
        wav_bytes: bytes | None = None,
        source: str = "live",
    ) -> None:
        """Overwrites the saved result for this video (rerun = cache refresh)."""
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {
            "glosses": glosses,
            "sentence": sentence,
            "duration": duration,
            "source": source,  # 'live' (web run) | 'prebake' (pre-run by a script)
            "saved_at": time.time(),
        }
        # Write-then-rename avoids a half-written JSON if the process dies mid-write.
        tmp = self._json_path(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._json_path(key))

        wav = self.wav_path(key)
        if wav_bytes:
            wav.write_bytes(wav_bytes)
        elif wav.is_file():
            wav.unlink()  # no voice this run -> drop the stale wav so it won't mismatch
        logger.info("Da luu cache %s (%d tu, source=%s).", key, len(glosses), source)
