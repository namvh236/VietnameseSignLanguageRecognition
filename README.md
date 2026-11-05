# SignSpeak

**Tiếng Việt** | [English](README.en.md)

**Hệ thống dịch ngôn ngữ ký hiệu tiếng Việt (VSL) thời gian thực** — nhận dạng cử chỉ từ
webcam hoặc video, ghép thành câu tiếng Việt tự nhiên và đọc thành giọng nói.

```
webcam / video  ──▶  keypoint  ──▶  nhận dạng ký hiệu  ──▶  ghép câu  ──▶  giọng nói
                   MediaPipe        SPOTER + SL-GCN         Qwen3        VieNeu-TTS
                   (trình duyệt)    (ONNX, CUDA)            (Ollama)
```

---

## Tính năng

| | |
|---|---|
| **Nhận dạng realtime** | Ký trực tiếp trước webcam, hệ thống tự tách từng ký hiệu theo động tác giơ/hạ tay |
| **Dịch video khi đang phát** | Tải video lên, ký hiệu được dịch ngay trong lúc video chạy — không phải chờ xử lý xong |
| **Ghép câu tự nhiên** | LLM chuyển chuỗi từ rời rạc thành câu tiếng Việt đúng ngữ pháp |
| **Đọc thành tiếng** | Tổng hợp giọng nói tiếng Việt, chọn được giọng nam/nữ và tốc độ |
| **Suy giảm mềm** | Thiếu Ollama hoặc TTS thì hệ thống vẫn nhận dạng bình thường, chỉ trả về từ ghép thô |

---

## Kiến trúc

| Thành phần | Vai trò |
|---|---|
| `backend/serving/` | FastAPI: WebSocket `/ws/keypoints` (luồng chính) + REST. Chạy ONNX trên CUDA |
| `backend/src/` | AI core: tiền xử lý keypoint theo đúng pipeline lúc huấn luyện |
| `backend/models/` | ONNX đã xuất: `spoter/spoter_v3.onnx`, `sl-gcn/sl_gcn_ensemble.onnx` |
| `frontend/` | React + Vite. MediaPipe Holistic chạy **trong trình duyệt**, chỉ gửi keypoint lên server |
| `backend/demo_gradio.py` | Bản demo Gradio độc lập, không cần frontend |

**Nhận dạng dùng late-fusion hai mô hình:** `P = 0.75 × P_slgcn + 0.25 × P_spoter`. Cả hai
ONNX đã bao gồm softmax trong đồ thị.

**Trích keypoint đặt ở client** để server không phải nhận luồng video: mỗi frame gửi đi chỉ
là 108 + 81 số thay vì cả khung hình.

---

## Yêu cầu

- GPU NVIDIA (CUDA) — ONNX Runtime chạy trên GPU
- Python 3.12 + [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- [Ollama](https://ollama.com) với model `qwen3:1.7b` *(tuỳ chọn — chỉ dùng cho bước ghép câu)*

---

## Cài đặt

```bash
# Backend (lần đầu tải vài GB: torch cu126, onnxruntime-gpu, ...)
cd backend && uv sync

# Frontend
cd frontend && npm install

# Model cho bước ghép câu
ollama pull qwen3:1.7b
```

Kiểm tra model ONNX và bảng nhãn đã có đủ:

```bash
cd backend && ./scripts/download_weights.sh
```

---

## Chạy

**Cách nhanh nhất** — một lệnh chạy cả backend + frontend (Ctrl+C để dừng tất cả):

```bash
./run_demo_web.sh
```

Script tự bật Ollama nếu có (native → Docker), log nằm ở `/tmp/signspeak/`. Đợi backend nạp
xong model (khoảng 20–30 giây) rồi mở **http://localhost:5173**.

**Chạy thủ công:**

```bash
cd backend  && uv run uvicorn serving.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

**Bản demo Gradio** (không cần frontend, mở `http://localhost:7860`):

```bash
./run_demo.sh
```

---

## Cấu hình

Toàn bộ cấu hình backend nằm trong `backend/serving/config.py` (pydantic-settings, ghi đè
được bằng biến môi trường).

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `fusion_w_slgcn` | `0.75` | Trọng số SL-GCN trong late-fusion |
| `realtime_fusion_confidence_threshold` | `0.30` | Ngưỡng tin cậy để chấp nhận một ký hiệu |
| `seg_angle_threshold` | `115.0` | Góc khuỷu tay coi là "đang giơ tay" (độ) |
| `seg_min_down_frames` | `12` | Số frame hạ tay liên tiếp để chốt kết thúc một ký hiệu |
| `ollama_model` | `qwen3:1.7b` | Model ghép câu |

Frontend đọc endpoint từ `frontend/.env.development`:

```
VITE_WS_URL=ws://localhost:8000/ws/keypoints
VITE_API_URL=http://localhost:8000
```

---

## Lưu ý khi sử dụng

**Cần hạ tay giữa các từ.** Mô hình nhận dạng ký hiệu rời rạc (isolated-sign), không phải
nhận dạng liên tục. Việc tách từ dựa vào động tác giơ/hạ tay, nên ký liền mạch không ngắt
quãng sẽ bị gộp thành một ký hiệu.

**Tốc độ phụ thuộc GPU của trình duyệt.** MediaPipe chạy bằng WebGL nên dùng GPU mà trình
duyệt chọn — thường là card tích hợp trên máy đồ hoạ lai. Giao diện hiển thị nhịp xử lý thực
tế (`AI x.x fps`); dưới ngưỡng an toàn sẽ có cảnh báo vì ký hiệu có thể bị bỏ sót.

**Lỗi WebGL.** Nếu gặp thông báo không dùng được WebGL, thường do quá nhiều tab đang chiếm
ngữ cảnh WebGL — đóng bớt tab rồi tải lại trang.

---

## Phát triển

```bash
cd backend  && make lint && make format    # flake8 + isort + black
cd frontend && npm run build               # kiểm tra build production
```

Mã nguồn phần huấn luyện, đánh giá và xuất ONNX nằm ở repo riêng; repo này chỉ tiêu thụ
model đã xuất.

Tài liệu chi tiết: [`backend/README.md`](backend/README.md) ·
[`frontend/README.md`](frontend/README.md)
