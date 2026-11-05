# SignSpeak Frontend

[Tiếng Việt](README.md) | **English**

Web UI (React + Vite) for the SignSpeak Vietnamese Sign Language (VSL) translation system.
It uses MediaPipe Holistic in the browser to extract keypoints from the webcam and sends them
over WebSocket to the FastAPI backend for sign recognition, sentence composition and speech.

```
webcam → MediaPipe Holistic (in-browser) → 108-float keypoints → ws://.../ws/keypoints
       ← gloss / Vietnamese sentence / audio (WAV bytes)
```

## Layout

```text
src/
├── App.jsx                 # layout, WebSocket setup
├── main.jsx                # entry point
├── components/
│   ├── CameraPanel/        # camera + frame sending
│   ├── ResultPanel/        # gloss / sentence / audio display
│   ├── ManualPanel/        # video upload / recording
│   └── LandingPage/
├── hooks/                  # useCamera
├── services/               # ws.js, holistic.js, videoSignStream.js, resultCache.js
├── store/                  # Zustand store (useSignStore)
└── utils/                  # keypoints, drawSkeleton, videoHash, audioHelper
```

## Install & run

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # production build
npm run lint      # oxlint
```

## Environment

Copy the matching `.env.development` or `.env.production`, or edit directly:

```bash
VITE_WS_URL=ws://localhost:8000/ws/keypoints
VITE_API_URL=http://localhost:8000
```

The serving backend must be running first (see `backend/README.md`).
