#!/usr/bin/env bash
# Dung tat ca tien trinh cua demo web SignSpeak (nguoc lai voi run_demo_web.sh).
#
# Chay:  ./stop_demo_web.sh
#
# KHONG dung Ollama service cua he thong (systemd) — no chay doc lap va app khac co the
# dang dung. Chi dong phien 'ollama run' tuong tac ma run_demo_web.sh mo ra.

set -uo pipefail

# Diet theo CONG, khong theo ten tien trinh: 'npm run dev' de ra mot chuoi tien trinh con
# (npm -> sh -c vite -> node .../vite) nen doan ten rat de truot. Cong thi luon dung.
stop_port() {
  local label="$1" port="$2"
  local pids
  pids="$(lsof -t -i :"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    echo "  $label (cong $port): khong chay"
    return
  fi
  echo "  $label (cong $port): dung PID $(echo "$pids" | tr '\n' ' ')"
  # Diet ca cay tien trinh: giet moi node/npm cha se de lai tien trinh con mo cong.
  # shellcheck disable=SC2086
  kill -- $pids 2>/dev/null
  sleep 1
  pids="$(lsof -t -i :"$port" -sTCP:LISTEN 2>/dev/null || true)"
  # shellcheck disable=SC2086
  [ -n "$pids" ] && kill -9 -- $pids 2>/dev/null
  return 0
}

echo "Dang dung demo web SignSpeak..."

# Tim tab qua BIEN MOI TRUONG trong /proc/<pid>/environ, khong qua dong lenh: tab chay
# `bash -c '...; exec bash'` nen khi lenh chinh ket thuc, dong lenh bi thay sach — chi moi
# truong da export la con lai.
find_tabs() {
  local pid
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    if grep -qz "^SIGNSPEAK_DEMO_TAB=" "/proc/$pid/environ" 2>/dev/null; then
      echo "$pid"
    fi
  done
}
TAB_PIDS="$(find_tabs)"

stop_port    "backend  (uvicorn)" 8000
stop_port    "frontend (vite)"    5173

# Dong luon cac tab do run_demo_web.sh mo. Tim theo token SIGNSPEAK_DEMO_TAB (duy nhat, do
# run script gan vao) chu KHONG theo pattern chung chung nhu 'bash -c ... npm run dev' —
# pattern rong kieu do tung khop nham va giet oan shell khac.
close_tabs() {
  if [ -z "$TAB_PIDS" ]; then
    echo "  tab terminal: khong co tab nao cua demo"
    return
  fi
  echo "  tab terminal: dong PID $(echo "$TAB_PIDS" | tr '\n' ' ')"
  # Bien moi truong duoc THUA KE, nen ca shell cua tab lan tien trinh con ('ollama run') deu
  # mang marker. Quet LAI sau moi luot diet thay vi tin vao danh sach chup mot lan: diet con
  # xong thi cha moi lo ra, va nguoc lai.
  local round pids pid
  for round in 1 2 3; do
    pids="$(find_tabs)"
    [ -z "$pids" ] && break
    for pid in $pids; do
      [ "$round" -eq 1 ] && kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
    done
    sleep 1
  done
  return 0
}
close_tabs

sleep 1
echo "Kiem tra lai:"
for port in 8000 5173; do
  if lsof -t -i :"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  cong $port: VAN CON MO"
  else
    echo "  cong $port: da dong"
  fi
done
echo "Xong." 
