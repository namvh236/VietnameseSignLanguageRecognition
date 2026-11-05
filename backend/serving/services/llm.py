"""Joins glosses into a Vietnamese sentence via Qwen3 (Ollama /api/chat); prompt +
/no_think + <think>-stripping ported from demo_gradio.py.
"""

import re

import httpx

from serving.config import settings
from serving.services.gloss_decoder import dedupe_consecutive
from serving.utils.logger import get_logger

logger = get_logger(__name__)

# NOTE: prompt text must keep Vietnamese diacritics -- an unaccented prompt makes the
# model produce worse sentences. (Only code comments go unaccented.)
SYSTEM_PROMPT = (
    "Bạn là trợ lý dịch ngôn ngữ ký hiệu tiếng Việt. Đầu vào là danh sách các từ (gloss) "
    "nhận dạng được từ người ký hiệu, theo đúng thứ tự thể hiện. Ngôn ngữ ký hiệu lược bỏ "
    "hư từ và chủ ngữ hiển nhiên, nên nhiệm vụ của bạn là khôi phục chúng để thành MỘT câu "
    "tiếng Việt nói tự nhiên, đúng ngữ pháp, lịch sự.\n"
    "QUY TẮC:\n"
    "1. Giữ ĐỦ tất cả các từ đã cho, ĐÚNG thứ tự — chúng là bộ khung của câu. Không bỏ, "
    "không đảo thứ tự.\n"
    "2. ĐƯỢC thêm hư từ, từ nối, tiểu từ tình thái: là, và, thì, mà, của, đang, rồi, một, "
    "ạ, ơi, với, nhé, nhỉ, không.\n"
    "3. ĐƯỢC thêm chủ ngữ hiển nhiên mà tiếng Việt bắt buộc phải có khi từ đã cho chỉ là "
    "một tính từ trạng thái. Thời tiết (nóng, lạnh, mưa, nắng, gió) cần chủ ngữ 'trời'.\n"
    "4. ĐƯỢC chuyển từ điển sang dạng nói tự nhiên: 'ngày' ở đầu câu → 'hôm nay'; từ chỉ "
    "người ở đầu một lời nhờ vả/xin phép → thêm 'ơi' để gọi (bác sĩ → 'bác sĩ ơi').\n"
    "5. Câu là lời nhờ vả, xin phép thì kết bằng 'với', 'ạ', 'nhé' cho tự nhiên.\n"
    "6. NGOÀI các phép thêm ở mục 2-5, KHÔNG thêm từ mang nghĩa mới: không thêm động từ, "
    "danh từ, tên người, địa điểm, thời gian nào không có trong danh sách. Không bịa nội dung.\n"
    "7. Dùng dấu phẩy tách hai mệnh đề nếu câu có hai ý. Viết hoa chữ đầu câu.\n"
    "8. Nếu chỉ có một từ, trả về đúng từ đó (viết hoa chữ đầu, thêm dấu câu).\n"
    "VÍ DỤ:\n"
    "- Từ: Ngày, Nóng, Em, Muốn, Uống, Nước → Hôm nay trời nóng, em muốn uống nước.\n"
    "- Từ: Bác sĩ, Khám bệnh, Cho, Cháu → Bác sĩ ơi, khám bệnh cho cháu với.\n"
    "- Từ: Ngày, Mưa, Em, Không, Đi, Học → Hôm nay trời mưa, em không đi học.\n"
    "- Từ: Cô, Giúp, Cho, Em → Cô ơi, giúp cho em với.\n"
    "- Từ: Tôi, Muốn, Ăn, Cơm → Tôi muốn ăn cơm.\n"
    "- Từ: Mẹ, Đi, Chợ → Mẹ đi chợ.\n"
    "- Từ: Anh, Khỏe, Không → Anh khỏe không?\n"
    "- Từ: Xin chào → Xin chào.\n"
    "Chỉ trả về câu kết quả, không giải thích, không dùng thẻ <think> hay định dạng đặc biệt."
)

# Ollama's first-time model load (cold start) can take longer than the usual timeout.
WARMUP_TIMEOUT = 120.0

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>")
# Qwen3 sometimes leaks the /think, /no_think control tokens into the output text.
_CONTROL_TOKEN_RE = re.compile(r"\s*/(?:no_)?think\b")
# A line that's just a model-inserted label before the sentence (e.g. "bản dịch").
_LABEL_ONLY_LINE_RE = re.compile(
    r"^(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：.\-–]*\s*$",
    flags=re.IGNORECASE,
)
# Label prefix at line start; stripped only with a trailing colon (a reliable signal).
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:bản dịch|câu(?:\s+dịch|\s+trả lời)?|kết quả|đáp án|trả lời|output|answer)"
    r"\s*[:：]\s*",
    flags=re.IGNORECASE,
)


def _strip_think(text: str) -> str:
    """Cleans LLM output: strips think blocks/tags, control tokens, stray labels."""
    text = _THINK_RE.sub("", text)
    text = _THINK_TAG_RE.sub("", text)
    text = _CONTROL_TOKEN_RE.sub("", text)
    cleaned: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _LABEL_ONLY_LINE_RE.match(line):
            continue
        line = _LABEL_PREFIX_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned).strip()


class LLMService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def load(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=settings.ollama_timeout
        )

    async def gloss_to_sentence(
        self, gloss_sequence: list[str], timeout=httpx.USE_CLIENT_DEFAULT
    ) -> str:
        gloss_sequence = dedupe_consecutive(gloss_sequence)
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    # Keep diacritics (see SYSTEM_PROMPT note): unaccented text here
                    # would skew the distribution vs. the accented system prompt.
                    "content": f"Các từ nhận dạng được: {', '.join(gloss_sequence)} /no_think",
                },
            ],
            # Disable reasoning at the API level (avoids <think> leaking); /no_think
            # in the prompt is a fallback for older Ollama builds without "think".
            "think": False,
            "options": {
                # Low temperature for stable output (constrained task, not creative).
                "temperature": 0.15,
                "num_predict": settings.ollama_num_predict,
            },
            "keep_alive": settings.ollama_keep_alive,
            "stream": False,
        }
        resp = await self._client.post("/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        sentence = _strip_think(content)
        return sentence or " ".join(gloss_sequence)

    async def ping(self) -> None:
        """Warms the model (cold start can be slow) so requests don't time out."""
        await self.gloss_to_sentence(["xin chao"], timeout=WARMUP_TIMEOUT)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
