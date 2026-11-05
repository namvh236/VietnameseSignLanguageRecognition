# SignSpeak

[Tiếng Việt](README.md) | **English**

**Real-time Vietnamese Sign Language (VSL) translation** — recognizes gestures from a
webcam or video, composes them into a natural Vietnamese sentence and reads it aloud.

```
webcam / video  ──▶  keypoints  ──▶  sign recognition  ──▶  sentence  ──▶  speech
                    MediaPipe       SPOTER + SL-GCN         Qwen3         VieNeu-TTS
                    (browser)       (ONNX, CUDA)            (Ollama)
```

---

## Features

| | |
|---|---|
| **Real-time recognition** | Sign live in front of a webcam; signs are segmented automatically from raising/lowering the hand |
| **Live video translation** | Upload a video and signs are translated while it plays — no waiting for processing to finish |
| **Natural sentences** | An LLM turns the sequence of isolated words into a grammatical Vietnamese sentence |
| **Text-to-speech** | Vietnamese speech synthesis with selectable male/female voice and speed |
| **Graceful degradation** | Without Ollama or TTS, recognition still works and returns the raw joined words |

---

## Architecture

| Component | Role |
|---|---|
| `backend/serving/` | FastAPI: WebSocket `/ws/keypoints` (main path) + REST. Runs ONNX on CUDA |
| `backend/src/` | AI core: keypoint preprocessing identical to the training pipeline |
| `backend/models/` | Exported ONNX: `spoter/spoter_v3.onnx`, `sl-gcn/sl_gcn_ensemble.onnx` |
| `frontend/` | React + Vite. MediaPipe Holistic runs **in the browser** and only keypoints are sent to the server |
| `backend/demo_gradio.py` | Standalone Gradio demo, no frontend needed |

**Recognition uses late fusion of two models:** `P = 0.75 × P_slgcn + 0.25 × P_spoter`. Both
ONNX graphs already include softmax.

**Keypoint extraction runs on the client** so the server never receives a video stream: each
frame sends only 108 + 81 numbers instead of the whole image.

---

## Requirements

- NVIDIA GPU (CUDA) — ONNX Runtime runs on the GPU
- Python 3.12 + [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- [Ollama](https://ollama.com) with the `qwen3:1.7b` model *(optional — only used for sentence composition)*

---

## Installation

```bash
# Backend (first run downloads several GB: torch cu126, onnxruntime-gpu, ...)
cd backend && uv sync

# Frontend
cd frontend && npm install

# Model for sentence composition
ollama pull qwen3:1.7b
```

Check that the ONNX models and the label table are in place:

```bash
cd backend && ./scripts/download_weights.sh
```

---

## Running

**Fastest way** — one command runs backend + frontend (Ctrl+C stops everything):

```bash
./run_demo_web.sh
```

The script starts Ollama if available (native → Docker); logs are in `/tmp/signspeak/`. Wait
for the backend to load the models (about 20–30 seconds), then open **http://localhost:5173**.

**Manually:**

```bash
cd backend  && uv run uvicorn serving.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

**Gradio demo** (no frontend needed, open `http://localhost:7860`):

```bash
./run_demo.sh
```

---

## Configuration

All backend settings live in `backend/serving/config.py` (pydantic-settings, overridable via
environment variables).

| Setting | Default | Meaning |
|---|---|---|
| `fusion_w_slgcn` | `0.75` | SL-GCN weight in the late fusion |
| `realtime_fusion_confidence_threshold` | `0.30` | Confidence threshold to accept a sign |
| `seg_angle_threshold` | `115.0` | Elbow angle treated as "hand raised" (degrees) |
| `seg_min_down_frames` | `12` | Consecutive hand-down frames needed to end a sign |
| `ollama_model` | `qwen3:1.7b` | Sentence composition model |

The frontend reads its endpoints from `frontend/.env.development`:

```
VITE_WS_URL=ws://localhost:8000/ws/keypoints
VITE_API_URL=http://localhost:8000
```

---

## Usage notes

**Lower your hands between words.** The model recognizes isolated signs, not continuous
signing. Segmentation relies on raising/lowering the hand, so signing without pauses gets
merged into a single sign.

**Speed depends on the browser's GPU.** MediaPipe runs on WebGL, so it uses whichever GPU the
browser picks — often the integrated one on hybrid-graphics laptops. The UI shows the actual
processing rate (`AI x.x fps`); below the safe threshold a warning appears because signs may
be missed.

**WebGL errors.** If you see a message that WebGL is unavailable, too many tabs are usually
holding WebGL contexts — close some tabs and reload the page.

---

## Development

```bash
cd backend  && make lint && make format    # flake8 + isort + black
cd frontend && npm run build               # production build check
```

Training, evaluation and ONNX export code lives in a separate repo; this repo only consumes
the exported models.

Detailed docs: [`backend/README.md`](backend/README.md) ·
[`frontend/README.md`](frontend/README.md)
