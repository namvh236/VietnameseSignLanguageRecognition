#!/usr/bin/env bash
# Check that model artifacts already exist under the project root (no download).
# Artifacts (ONNX + gloss.csv) live next to serving; see serving/config.py.
set -euo pipefail

# scripts/ -> .. = project root (backend/), holding models/ and experiments/.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="${RECOGNITION_ROOT:-$ROOT}"

SPOTER_ONNX="$PROJECT_ROOT/models/spoter/spoter_v3.onnx"
SLGCN_ONNX="$PROJECT_ROOT/models/sl-gcn/sl_gcn_ensemble.onnx"
GLOSS="$PROJECT_ROOT/experiments/gloss.csv"

missing=0
for f in "$SPOTER_ONNX" "$SLGCN_ONNX" "$GLOSS"; do
  if [[ -f "$f" ]]; then
    echo "OK  $f"
  else
    echo "THIEU  $f"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Thieu artifact. Xem README.md de tao/export model." >&2
  exit 1
fi
echo "Day du artifact."
