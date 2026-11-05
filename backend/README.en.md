# SignSpeak backend — VSL-400 serving

[Tiếng Việt](README.md) | **English**

The **serving** side of the Vietnamese Sign Language (VSL) translation system:
FastAPI + a Gradio demo running two ONNX models (SPOTER + SL-GCN late fusion), sentence
composition with Qwen3 (Ollama) and speech with VieNeu-TTS.

> Training / evaluation / ONNX export code lives in a **separate training repo** — this repo
> only consumes the exported artifacts.

## Layout

```text
pyproject.toml               # uv-managed (Python 3.12, torch cu126)
demo_gradio.py               # all-in-one Gradio demo (no frontend needed)
serving/                     # FastAPI: WS /ws/keypoints + REST
src/
  ├── pipelines/spoter_graph_classification.py   # SPOTER preprocessing transforms
  └── utils/constants.py                         # SLGCN_JOINTS, COCO_TO_POSE_FORMAT
scripts/                     # download_weights.sh, eval_segmentation.py, prebake_videos.py
models/
  ├── spoter/spoter_v3.onnx                      # (git-untracked)
  └── sl-gcn/sl_gcn_ensemble.onnx                # (git-untracked)
experiments/gloss.csv        # 400 gloss labels (id,gloss — git-untracked)
```

## Install & run

```bash
uv sync                                                     # needs an NVIDIA GPU (torch cu126)
./scripts/download_weights.sh                               # checks that the artifacts exist
uv run uvicorn serving.main:app --host 0.0.0.0 --port 8000  # serving for the React frontend
uv run python demo_gradio.py                                # or the Gradio demo on :7860
```

Sentence composition needs Ollama at `OLLAMA_BASE_URL` (default `http://localhost:11434`)
with the model `OLLAMA_MODEL` (default `qwen3:1.7b`). Without Ollama, recognition and TTS
still work and the sentence is the raw joined glosses.

### Realtime keypoint contract (108 floats / frame)

The client (frontend) sends keypoints, already extracted by MediaPipe Holistic and joint-selected,
over `/ws/keypoints`: 54 joints × (x, y) in **pixel coordinates**, in `JointSelect` order:
12 body joints (nose, neck, rightEye, leftEye, rightEar, leftEar, rightShoulder, leftShoulder,
rightElbow, leftElbow, rightWrist, leftWrist), then 21 LEFT-hand joints, then 21 RIGHT-hand
joints. **Joint index 1 (neck) must be `[0, 0]`.**

## API

- `WS /ws/keypoints` — accepts `{"keypoints": [108 floats], "end_sign": bool}`; returns JSON
  `gloss`/`processing`/`sentence`/`error` and a **binary WAV** after `sentence`.
- `POST /api/recognize` — one-shot fallback: `{"frames": [[108 floats], ...]}`.
- `POST /api/recognize-video` — accepts a video file and runs MediaPipe Holistic + SPOTER/SL-GCN
  late fusion itself (same as `demo_gradio.py`); returns glosses only.
- `POST /api/compose` — composes a list of glosses into a sentence + audio.
- `POST /api/tts` — `{"text": "..."}` → `audio/wav` bytes.
- `GET /api/cache/{sha256}`, `POST /api/cache/{sha256}` — result cache keyed by video content hash.
- `GET /health` — spoter/ollama/tts status.
