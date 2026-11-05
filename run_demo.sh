#!/usr/bin/env bash
# Run the Gradio demo on :7860 (starts Ollama first if available).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-7860}"
URL="http://localhost:${PORT}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"

# A stale VIRTUAL_ENV makes uv use the wrong venv.
unset VIRTUAL_ENV 2>/dev/null || true

# shellcheck source=scripts/ollama.sh
source "$ROOT_DIR/scripts/ollama.sh"

echo "🤟 SignSpeak — khởi chạy web demo"
ensure_ollama

echo "→ Mở web demo tại ${URL}  (nhấn Ctrl+C để dừng)"
if have xdg-open; then
  ( sleep 5; xdg-open "${URL}" >/dev/null 2>&1 || true ) &
fi

cd "$ROOT_DIR/backend"
exec uv run python demo_gradio.py
