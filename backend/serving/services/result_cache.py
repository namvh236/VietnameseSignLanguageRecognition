"""Cache ket qua dich theo TUNG FILE video (an toan cho demo).

Van de: moi lan tai lai cung mot video, ca duong ong (MediaPipe -> ONNX -> LLM -> TTS) phai
chay lai tu dau — cham va co the loi ngay tren san khau. Cache nay luu ket qua da chay dung
cua mot video, khoa theo NOI DUNG file (sha256 do client tinh), nen tai lai video cu la phat
lai ket qua cu tuc thi.

Layout (git-untracked):
    cache/results/<key>.json   {glosses: [{text, at}], sentence, duration, source, saved_at}
    cache/results/<key>.wav    WAV cua TTS (khong co neu luc luu TTS loi)

Key den tu client -> PHAI kiem tra dinh dang truoc khi ghep vao duong dan (chong path
traversal). Chi cho phep hex.
"""

import json
import re
import time
from pathlib import Path

from serving.utils.logger import get_logger

logger = get_logger(__name__)

# Client uu tien sha256 (64 hex); moi truong khong co crypto.subtle (http tren LAN) lui ve
# khoa ngan hon tu ten+kich thuoc+thoi diem sua file.
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
        """Ket qua da luu (kem wav_bytes neu co), None neu chua tung xu ly."""
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
        """Ghi de ket qua cho video nay (chay lai = cap nhat cache)."""
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {
            "glosses": glosses,
            "sentence": sentence,
            "duration": duration,
            "source": source,  # 'live' (chay that tren web) | 'prebake' (script chuan bi truoc)
            "saved_at": time.time(),
        }
        # Ghi tam roi doi ten: tranh de lai file JSON do dang neu tien trinh chet giua chung.
        tmp = self._json_path(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._json_path(key))

        wav = self.wav_path(key)
        if wav_bytes:
            wav.write_bytes(wav_bytes)
        elif wav.is_file():
            wav.unlink()  # lan chay moi khong co giong noi -> bo wav cu cho khoi lech cau
        logger.info("Da luu cache %s (%d tu, source=%s).", key, len(glosses), source)
