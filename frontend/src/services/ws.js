import { blobToAudioUrl } from '../utils/audioHelper'

const WS_URL = import.meta.env.VITE_WS_URL

class WebSocketClient {
  constructor() {
    this.ws = null
    this.handlers = {}
    this.reconnectTimer = null
    this.shouldReconnect = false
    this.reconnectAttempts = 0
    // state for idle-finalize (auto-ending a sentence after inactivity)
    this.lastGlossAt = 0
    this.hasPending = false // true when a gloss hasn't been folded into a sentence yet
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
      // binary frame = WAV audio from TTS
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
          // backend finished the session (even if TTS failed) -> exit the waiting state
          case 'done':
            this.hasPending = false
            this.handlers.onProcessing?.(false)
            this.handlers.onFinished?.({ noSign: false })
            break
          // session ended with no confident sign -> no sentence, no audio
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

  /** Sends one keypoint frame; endSign=true tells the backend to build the sentence and speak it. Optional slgcnKeypoints (81 numbers) enables SL-GCN late fusion; omit it to run SPOTER-only. */
  sendKeypoints(keypoints, slgcnKeypoints = null, endSign = false) {
    if (this.ws?.readyState !== WebSocket.OPEN) return

    // Hold-last resends the same array repeatedly -> cache the JSON string instead of
    // re-serializing ~200 numbers each time. Reference equality is enough: new frames are new arrays.
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

  /** True when there's a pending gloss and hands have been idle for idleMs -> should finalize. */
  shouldFinalize(idleMs) {
    return this.hasPending && Date.now() - this.lastGlossAt >= idleMs
  }

  /** Clears client-side pending-gloss state; needed after an abnormal session end, or the next realtime session would finalize immediately on the first camera frame. */
  resetFinalizeState() {
    this.lastGlossAt = 0
    this.hasPending = false
  }

  /** Reopens the connection so the backend issues a fresh SessionState (one per connection); used after a mid-run cancel so the next sentence doesn't inherit leftover words. */
  resetSession() {
    this.resetFinalizeState()
    const old = this.ws
    if (!old) return
    this.ws = null
    // clear onclose before closing, or _handleReconnect will schedule a reconnect in 3s
    old.onclose = null
    try {
      old.close()
    } catch {
      // already closed -> fine
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
