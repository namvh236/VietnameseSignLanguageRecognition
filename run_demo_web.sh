#!/usr/bin/env bash
# Mo 3 tab terminal cho demo web SignSpeak:
#   1. backend  (FastAPI + ONNX tren GPU)
#   2. frontend (Vite dev server)
#   3. ollama   (qwen3:1.7b — buoc ghep cau)
#
# Chay:  ./run_demo_web.sh
# Dung:  ./stop_demo_web.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "Khong tim thay gnome-terminal." >&2
  exit 1
fi

# gnome-terminal coi MOI THU sau '--' la cau lenh, nen khong the xau chuoi nhieu '--tab ... --'
# trong cung mot lan goi (cac tab sau se bi nuot). Goi rieng tung lan: lan dau tao cua so,
# cac lan sau '--tab' tu gan vao cua so do.
# Danh dau tab bang bien MOI TRUONG da export, khong phai chuoi trong dong lenh: khi lenh
# chinh ket thuc (hoac bi Ctrl+C), `exec bash` thay the tien trinh va dong lenh MAT sach —
# nhung moi truong da export thi con nguyen. stop_demo_web.sh doc /proc/<pid>/environ de tim.
open_tab() {
  local title="$1" dir="$2" cmd="$3"
  # 'exec bash' de tab khong dong sap khi Ctrl+C — con xem duoc log va chay lai.
  gnome-terminal --tab --title="$title" --working-directory="$dir" \
    -- bash -c "export SIGNSPEAK_DEMO_TAB=$title; $cmd; exec bash"
  sleep 0.7   # cho cua so kip san sang de tab sau gan dung cho
}

# VIRTUAL_ENV con sot lai se lam 'uv run' tro nham venv -> phai xoa trong tab backend.
open_tab "backend"  "$ROOT/backend"  'unset VIRTUAL_ENV; uv run uvicorn serving.main:app --host 0.0.0.0 --port 8000'
open_tab "frontend" "$ROOT/frontend" 'npm run dev'
open_tab "ollama"   "$ROOT"          "ollama run $MODEL"

echo "Da mo 3 tab: backend / frontend / ollama ($MODEL)."
echo "Doi backend nap xong model roi mo: http://localhost:5173"
