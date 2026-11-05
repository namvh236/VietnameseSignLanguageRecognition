# Shared Ollama helpers, sourced by run_demo.sh and run_demo_web.sh.
# Caller must set ROOT_DIR, OLLAMA_URL and OLLAMA_MODEL.
# Sets OLLAMA_SERVE_PID when it starts a native `ollama serve`.

have() { command -v "$1" >/dev/null 2>&1; }

ollama_ready() {
  have curl && curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1
}

# Wait up to ~30s for Ollama to answer.
wait_ollama() {
  have curl || return 0
  for _ in $(seq 1 30); do ollama_ready && return 0; sleep 1; done
  return 1
}

# Order: already running -> native -> Docker -> skip (raw gloss join).
ensure_ollama() {
  local pull=""
  if ollama_ready; then
    echo "→ Ollama đã chạy (${OLLAMA_URL})."
    have ollama && pull="ollama"
  elif have ollama; then
    echo "→ Ollama native chưa chạy, đang bật 'ollama serve' ở nền..."
    ollama serve >/tmp/ollama-serve.log 2>&1 &
    OLLAMA_SERVE_PID=$!
    wait_ollama && echo "→ Ollama đã sẵn sàng." \
      || echo "⚠️  Ollama chưa lên — demo vẫn chạy, câu sẽ là ghép từ thô."
    pull="ollama"
  elif have docker; then
    echo "→ Không có Ollama native, dùng Docker: đang bật container..."
    docker compose -f "$ROOT_DIR/docker-compose.yml" up -d ollama \
      || echo "⚠️  Không bật được Ollama — demo vẫn chạy, câu sẽ là ghép từ thô."
    wait_ollama && echo "→ Ollama đã sẵn sàng." || true
    pull="docker compose -f $ROOT_DIR/docker-compose.yml exec -T ollama ollama"
  else
    echo "→ Không có Ollama lẫn Docker — bỏ qua LLM (câu = ghép từ thô). Nhận dạng + TTS vẫn chạy."
  fi

  [ -n "$pull" ] && ollama_ready || return 0
  if curl -sf "${OLLAMA_URL}/api/tags" 2>/dev/null | grep -q "\"${OLLAMA_MODEL}\""; then
    echo "→ Model ${OLLAMA_MODEL} đã sẵn sàng."
  else
    echo "→ Chưa có model ${OLLAMA_MODEL}, đang tải (lần đầu, có thể lâu)..."
    $pull pull "${OLLAMA_MODEL}" \
      || echo "⚠️  Không tải được model — câu sẽ là ghép từ thô."
  fi
}
