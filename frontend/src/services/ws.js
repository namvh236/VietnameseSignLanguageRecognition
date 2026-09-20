import { blobToAudioUrl } from '../utils/audioHelper'

const WS_URL = import.meta.env.VITE_WS_URL

class WebSocketClient {
  constructor() {
    this.ws = null
    this.handlers = {}
    this.reconnectTimer = null
    this.shouldReconnect = false
    this.reconnectAttempts = 0
    // Trạng thái để tự động chốt câu khi nghỉ tay (idle finalize).
    this.lastGlossAt = 0
    this.hasPending = false // có gloss chưa được ghép câu
  }

  connect({ onStatus, onGloss, onSentence, onAudio, onProcessing, onFinished }) {
    this.handlers = { onStatus, onGloss, onSentence, onAudio, onProcessing, onFinished }
    this.shouldReconnect = true
    this.reconnectAttempts = 0 // Reset attempts on fresh connect
    this._open()
  }

  _open() {
    if (this.reconnectAttempts >= 5) {
      console.warn(`WebSocket connection failed after 5 attempts. Stopping reconnect.`)
      this.shouldReconnect = false
      this.handlers.onStatus?.('disconnected')
      return
    }

    this.reconnectAttempts++
    this.handlers.onStatus?.('connecting')

    try {
      this.ws = new WebSocket(WS_URL)
    } catch (e) {
      console.error('WebSocket connection initialization failed:', e)
      this.handlers.onStatus?.('error')
      this._handleReconnect()
      return
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0 // Reset attempts on successful connection
      this.handlers.onStatus?.('connected')
    }

    this.ws.onclose = () => {
      // If we intentionally stopped reconnecting, don't set status to error or reconnect
      if (this.shouldReconnect) {
        this.handlers.onStatus?.('disconnected')
        this._handleReconnect()
      }
    }

    this.ws.onerror = () => {
      if (this.shouldReconnect) {
        this.handlers.onStatus?.('error')
      }
    }

    this.ws.onmessage = (event) => {
      // Khung nhị phân = WAV từ TTS.
      if (event.data instanceof Blob) {
        const blobUrl = blobToAudioUrl(event.data)
        this.handlers.onAudio?.(blobUrl)
        this.handlers.onProcessing?.(false)
        this.hasPending = false
        return
      }

      try {
        const data = JSON.parse(event.data)
        switch (data.type) {
          case 'gloss':
            this.lastGlossAt = Date.now()
            this.hasPending = true
            this.handlers.onGloss?.(data.gloss)
            break
          case 'processing':
            this.handlers.onProcessing?.(true)
            break
          case 'sentence':
            this.handlers.onSentence?.(data.text)
            break
          // Backend đã chốt xong phiên (kể cả khi TTS lỗi) -> thoát trạng thái chờ.
          case 'done':
            this.hasPending = false
            this.handlers.onProcessing?.(false)
            this.handlers.onFinished?.({ noSign: false })
            break
          // Hết phiên mà không ký hiệu nào đủ tin cậy -> không có câu, không có giọng nói.
          case 'no_sign':
            this.hasPending = false
            this.handlers.onProcessing?.(false)
            this.handlers.onFinished?.({ noSign: true })
            break
          case 'error':
            console.error('Backend error:', data.error)
            this.handlers.onProcessing?.(false)
            this.handlers.onFinished?.({ noSign: false })
            break
          default:
            console.warn('Unknown message type received:', data.type)
        }
      } catch (err) {
        console.error('Error parsing WS message:', err)
      }
    }
  }

  _handleReconnect() {
    if (this.shouldReconnect) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = setTimeout(() => this._open(), 3000)
    }
  }

  /**
   * Gửi một frame keypoint. endSign=true để backend ghép câu + đọc.
   * slgcnKeypoints (tuỳ chọn, 81 số) cho late-fusion SL-GCN; thiếu -> backend chạy SPOTER-only.
   */
  sendKeypoints(keypoints, slgcnKeypoints = null, endSign = false) {
    if (this.ws?.readyState !== WebSocket.OPEN) return

    // Hold-last gửi lại cùng một mảng nhiều lần liên tiếp -> nhớ chuỗi JSON đã tạo thay vì
    // serialize gần 200 số mỗi lần. So sánh tham chiếu là đủ: frame mới luôn là mảng mới.
    let payload
    if (!endSign && keypoints === this._lastKeypoints && slgcnKeypoints === this._lastSlgcn) {
      payload = this._lastPayload
    } else {
      const obj = { keypoints, end_sign: endSign }
      if (slgcnKeypoints) obj.slgcn_keypoints = slgcnKeypoints
      payload = JSON.stringify(obj)
      if (!endSign) {
        this._lastKeypoints = keypoints
        this._lastSlgcn = slgcnKeypoints
        this._lastPayload = payload
      }
    }

    this.ws.send(payload)
    if (endSign) this.hasPending = false
  }

  /**
   * True khi có gloss mới và đã nghỉ tay đủ lâu (idleMs) -> nên chốt câu.
   */
  shouldFinalize(idleMs) {
    return this.hasPending && Date.now() - this.lastGlossAt >= idleMs
  }

  /**
   * Quên các gloss đang chờ chốt câu (phía client). Cần khi phiên kết thúc bất thường, nếu
   * không lần sau vào tab realtime sẽ tự chốt câu ngay từ frame camera đầu tiên.
   */
  resetFinalizeState() {
    this.lastGlossAt = 0
    this.hasPending = false
  }

  /**
   * Mở lại kết nối để backend cấp SessionState mới (mỗi kết nối = một phiên độc lập).
   * Dùng khi huỷ giữa chừng, tránh câu lần sau dính từ của lần chạy bị huỷ.
   */
  resetSession() {
    this.resetFinalizeState()
    const old = this.ws
    if (!old) return
    this.ws = null
    // Gỡ onclose trước khi đóng, nếu không _handleReconnect sẽ hẹn kết nối lại sau 3s.
    old.onclose = null
    try {
      old.close()
    } catch {
      // Socket đã đóng sẵn -> không sao.
    }
    this.reconnectAttempts = 0
    this._open()
  }

  disconnect() {
    this.shouldReconnect = false
    clearTimeout(this.reconnectTimer)
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}

export default new WebSocketClient()
