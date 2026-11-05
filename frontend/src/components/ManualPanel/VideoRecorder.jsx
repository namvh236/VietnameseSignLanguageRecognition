import { useCallback, useEffect, useRef, useState } from 'react'
import { Circle, Square, Loader2, VideoOff } from 'lucide-react'

/** Records via webcam then hands off to UploadPlayer (same pipeline: MediaPipe -> keypoints -> WebSocket).
 * Preview is mirrored for the signer, but the saved file is NOT flipped — MediaPipe reads raw frames, matching training data. */

const MAX_MS = 60000 // auto-stop after 60s to cap clip length
const MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4',
]

/** Codec support varies by browser; pick the first one that works. */
function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return null
  for (const mime of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime
  }
  return '' // let the browser choose
}

function formatTime(ms) {
  const total = Math.floor(ms / 1000)
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export default function VideoRecorder({ onRecorded }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const startedAtRef = useRef(0)
  const timerRef = useRef(null)
  const autoStopRef = useRef(null)
  // Session token: bumping it lets a pending getUserMedia resolve after unmount be
  // discarded immediately instead of leaving the camera stuck on.
  const sessionRef = useRef(0)

  const [isRecording, setIsRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')

  // Open the camera once for the component's lifetime.
  useEffect(() => {
    const session = ++sessionRef.current
    let localStream = null

    const open = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Trình duyệt không hỗ trợ truy cập camera.')
        }
        localStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: false,
        })
        if (session !== sessionRef.current) {
          localStream.getTracks().forEach(t => t.stop())
          return
        }
        streamRef.current = localStream
        if (videoRef.current) {
          videoRef.current.srcObject = localStream
          try {
            await videoRef.current.play()
          } catch (playErr) {
            if (playErr.name !== 'AbortError') throw playErr
          }
        }
        setReady(true)
      } catch (err) {
        if (session !== sessionRef.current) return
        console.error('Lỗi mở camera để quay:', err)
        setError(
          err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError'
            ? 'Vui lòng cấp quyền truy cập camera để quay video.'
            : `Không thể mở camera: ${err.message}`
        )
      }
    }

    open()

    return () => {
      sessionRef.current++
      clearInterval(timerRef.current)
      clearTimeout(autoStopRef.current)
      const rec = recorderRef.current
      if (rec && rec.state !== 'inactive') {
        rec.onstop = null // unmounted mid-recording -> discard the half-finished clip
        try {
          rec.stop()
        } catch {
          // recorder already died with the stream, nothing to do
        }
      }
      streamRef.current?.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
  }, [])

  const stop = useCallback(() => {
    clearTimeout(autoStopRef.current)
    clearInterval(timerRef.current)
    const rec = recorderRef.current
    if (rec && rec.state !== 'inactive') rec.stop()
  }, [])

  const start = () => {
    const stream = streamRef.current
    if (!stream || isRecording) return

    const mime = pickMimeType()
    if (mime === null) {
      setError('Trình duyệt không hỗ trợ ghi video (MediaRecorder).')
      return
    }

    let recorder
    try {
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
    } catch (err) {
      console.error('Không tạo được MediaRecorder:', err)
      setError('Không bắt đầu quay được: ' + err.message)
      return
    }

    chunksRef.current = []
    recorderRef.current = recorder
    setError('')

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = () => {
      clearInterval(timerRef.current)
      clearTimeout(autoStopRef.current)
      setIsRecording(false)
      // Measure via wall clock: MediaRecorder's webm reports duration = Infinity, but
      // UploadPlayer's progress bar needs a real number.
      const duration = (performance.now() - startedAtRef.current) / 1000
      const type = recorder.mimeType || mime || 'video/webm'
      const blob = new Blob(chunksRef.current, { type })
      chunksRef.current = []
      if (blob.size === 0) {
        setError('Clip rỗng — hãy thử quay lại.')
        return
      }
      const ext = type.includes('mp4') ? 'mp4' : 'webm'
      const stamp = new Date()
      const name = `quay-${stamp.toTimeString().slice(0, 8).replace(/:/g, '-')}.${ext}`
      onRecorded({
        id: `record-${Date.now()}`,
        file: new File([blob], name, { type }),
        name,
        url: URL.createObjectURL(blob),
        duration,
      })
    }

    recorder.onerror = (e) => {
      console.error('Lỗi MediaRecorder:', e.error)
      setError('Lỗi khi quay: ' + (e.error?.message || 'không rõ nguyên nhân'))
      stop()
    }

    startedAtRef.current = performance.now()
    setElapsed(0)
    setIsRecording(true)
    recorder.start(200) // fixed chunk interval avoids losing data if the tab freezes
    timerRef.current = setInterval(
      () => setElapsed(performance.now() - startedAtRef.current),
      200
    )
    autoStopRef.current = setTimeout(stop, MAX_MS)
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col rounded-2xl border border-slate-200 dark:border-slate-900 bg-white/40 dark:bg-slate-950/25 overflow-hidden">
      <div className="relative flex-1 min-h-0 bg-slate-950 flex items-center justify-center">
        <video
          ref={videoRef}
          className="w-full h-full object-contain scale-x-[-1]"
          muted
          playsInline
        />

        {!ready && !error && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-slate-950/70 backdrop-blur-sm">
            <Loader2 size={28} className="animate-spin text-indigo-400" />
            <p className="text-xs font-semibold text-slate-200">Đang mở camera...</p>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-slate-950/80 px-8 text-center">
            <VideoOff size={28} className="text-rose-400" />
            <p className="text-xs font-semibold text-rose-300">{error}</p>
          </div>
        )}

        {isRecording && (
          <div className="absolute top-3 left-3 z-20 flex items-center gap-2 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-rose-500/40 text-[10px] font-bold text-slate-200 uppercase tracking-wider">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500" />
            </span>
            <span>Đang quay · {formatTime(elapsed)}</span>
          </div>
        )}
      </div>

      <div className="shrink-0 flex items-center justify-between gap-3 px-4 py-2.5 border-t border-slate-200 dark:border-slate-900">
        <p className="text-[11px] text-slate-500 dark:text-slate-400 min-w-0 truncate">
          {isRecording
            ? `Ký hiệu trước camera, hạ tay giữa các từ · tối đa ${MAX_MS / 1000}s`
            : 'Bấm quay, ký hiệu xong bấm dừng — clip sẽ được dịch ngay sau đó'}
        </p>

        {isRecording ? (
          <button
            onClick={stop}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-rose-500 hover:bg-rose-600 text-xs font-bold text-white shadow-md shadow-rose-500/20 transition-all shrink-0"
          >
            <Square size={13} className="fill-white" />
            <span>Dừng quay</span>
          </button>
        ) : (
          <button
            onClick={start}
            disabled={!ready}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed text-xs font-bold text-white shadow-md shadow-indigo-500/20 transition-all shrink-0"
          >
            <Circle size={13} className="fill-white" />
            <span>Bắt đầu quay</span>
          </button>
        )}
      </div>
    </div>
  )
}
