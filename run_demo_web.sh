#!/usr/bin/env bash
# Run backend (:8000) + frontend (:5173) in one terminal; Ctrl+C stops everything.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/signspeak}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"

# A stale VIRTUAL_ENV makes uv use the wrong venv.
unset VIRTUAL_ENV 2>/dev/null || true

# shellcheck source=scripts/ollama.sh
source "$ROOT_DIR/scripts/ollama.sh"

PIDS=()
TAIL_PID=""
OLLAMA_SERVE_PID=""

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "→ Đang dừng demo..."
  local pid
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null
  [ -n "$OLLAMA_SERVE_PID" ] && kill "$OLLAMA_SERVE_PID" 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap cleanup INT TERM EXIT

# setsid gives each service its own process group, so cleanup also kills
# children (uv -> uvicorn, npm -> vite).
start() {
  local name="$1" dir="$2"; shift 2
  (cd "$dir" && exec setsid "$@" >"$LOG_DIR/$name.log" 2>&1) &
  PIDS+=("$!")
}

mkdir -p "$LOG_DIR"
ensure_ollama

start backend  "$ROOT_DIR/backend"  uv run uvicorn serving.main:app --host 0.0.0.0 --port 8000
start frontend "$ROOT_DIR/frontend" npm run dev

echo "→ Backend: http://localhost:8000 | Frontend: http://localhost:5173"
echo "→ Đợi backend nạp model (~20–30s). Log: ${LOG_DIR}/{backend,frontend}.log. Ctrl+C để dừng."

tail -n 0 -F "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" &
TAIL_PID=$!
wait -n "${PIDS[@]}"
echo "⚠️  Một service đã thoát — xem log trong ${LOG_DIR}."
