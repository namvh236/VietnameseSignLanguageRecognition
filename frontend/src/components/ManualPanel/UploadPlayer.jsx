import { useRef, useState, useEffect, useCallback } from 'react'
import { Play, Square, Activity, Loader2, AlertTriangle } from 'lucide-react'
import useSignStore from '../../store/useSignStore'
import { onceEvent, streamVideoThroughWs } from '../../services/videoSignStream'
import wsClient from '../../services/ws'
import { drawSkeleton } from '../../utils/drawSkeleton'
import { base64ToAudioUrl } from '../../utils/audioHelper'
import {
  blobUrlToBase64,
  fetchCachedResult,
  saveCachedResult,
} from '../../services/resultCache'

/** Plays an uploaded video and translates it live, same pipeline as the realtime camera (video -> MediaPipe -> WebSocket); results go to source 'upload'. Not mirrored like the camera view — that's only for signer self-view and would reverse pre-recorded footage. */

/** Waits for backend finalize after video ends (end_sign must still flush final sign -> LLM -> TTS);
 * returning wsSink too early misroutes the last word/sentence/audio. Signal is finishedAt (always sent). */
function waitForUploadFinalize(source, overallMs = 25000, graceAfterSentenceMs = 6000) {
  return new Promise((resolve) => {
    const startedFinishedAt = useSignStore.getState().sources[source].finishedAt
    let unsub = () => {}
    let graceTimer = null
    const done = () => {
      clearTimeout(overall)
      clearTimeout(graceTimer)
      unsub()
      resolve()
    }
    const overall = setTimeout(done, overallMs)
    unsub = useSignStore.subscribe((s) => {
      const r = s.sources[source]
      if (r.finishedAt !== startedFinishedAt) return done() // backend signaled done
      if (r.audioUrl) return done() // audio present = fully done
      // sentence present but no audio yet (TTS slow/failed) -> wait a bit more
      if (r.sentence && !graceTimer) graceTimer = setTimeout(done, graceAfterSentenceMs)
    })
  })
}

// Simulated delay before showing the sentence on replay (real runs take about this long for LLM+TTS).
const REPLAY_FINALIZE_MS = 1200

export default function UploadPlayer({ item, source = 'upload', autoStart = false }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const cancelRef = useRef(false)
  const autoStartedRef = useRef(false)

  const [isRunning, setIsRunning] = useState(false)
  const [phase, setPhase] = useState(null) // 'warmup' | 'playing' | 'finalizing'
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [stats, setStats] = useState(null)
  // Cached result for this exact file (if processed before, or prebaked by script).
  const [cached, setCached] = useState(null)

  const showSkeleton = useSignStore(s => s.showSkeleton)
  const toggleSkeleton = useSignStore(s => s.toggleSkeleton)
  const setWsSink = useSignStore(s => s.setWsSink)
  const setActiveSource = useSignStore(s => s.setActiveSource)
  const startNewRun = useSignStore(s => s.startNewRun)
  const addGloss = useSignStore(s => s.addGloss)
  const setSentence = useSignStore(s => s.setSentence)
  const setAudioUrl = useSignStore(s => s.setAudioUrl)
  const markFinished = useSignStore(s => s.markFinished)
  const wsStatus = useSignStore(s => s.wsStatus)
  const noSign = useSignStore(s => s.sources[source].noSign)

  // Unmounting mid-run -> stop the loop and hand WS back to the realtime camera.
  useEffect(() => {
    return () => {
      cancelRef.current = true
      useSignStore.getState().setWsSink('realtime')
    }
  }, [])

  // Once cache key (file hash) is ready, ask backend if this video was already processed.
  useEffect(() => {
    let alive = true
    if (!item.cacheKey) return setCached(null)
    fetchCachedResult(item.cacheKey).then((res) => {
      if (alive) setCached(res)
    })
    return () => {
      alive = false
    }
  }, [item.cacheKey])

  const handleResults = useCallback((results) => {
    const video = videoRef.current
    if (!video || !showSkeleton || !canvasRef.current) return
    drawSkeleton(canvasRef.current, results, video.videoWidth, video.videoHeight)
  }, [showSkeleton])

  const stop = () => {
    cancelRef.current = true
  }

  /** Replays a cached result: plays video, emits words at recorded timestamps, then shows sentence+audio.
   * Skips MediaPipe/WebSocket/ONNX/LLM/TTS entirely. Deliberately indistinguishable in the UI from a live run. */
  const replayCached = async (entry) => {
    const video = videoRef.current
    if (!video || isRunning) return

    cancelRef.current = false
    setIsRunning(true)
    setError('')
    setProgress(0)
    setStats(null)
    setPhase('playing')

    startNewRun(source)
    setActiveSource(source)

    const timeline = [...(entry.glosses || [])].sort((a, b) => (a.at || 0) - (b.at || 0))
    let next = 0

    try {
      if (video.readyState < 1) await onceEvent(video, 'loadedmetadata', 5000)
      video.currentTime = 0
      await video.play()

      await new Promise((resolve) => {
        const total = Number.isFinite(video.duration) && video.duration > 0
          ? video.duration
          : entry.duration || item.duration || 0
        const timer = setInterval(() => {
          if (cancelRef.current) {
            clearInterval(timer)
            video.pause()
            return resolve()
          }
          const t = video.currentTime
          while (next < timeline.length && (timeline[next].at || 0) <= t) {
            addGloss(source, timeline[next].text)
            next++
          }
          setProgress(total > 0 ? Math.min(t / total, 1) : 0)
          if (video.ended) {
            clearInterval(timer)
            resolve()
          }
        }, 100)
      })

      if (cancelRef.current) return

      // Video may be shorter than the last timestamp (prebake marks are approximate) -> flush the rest.
      for (; next < timeline.length; next++) addGloss(source, timeline[next].text)

      setProgress(1)
      // Real runs take a few seconds for LLM+TTS; showing the sentence instantly would look off.
      setPhase('finalizing')
      await new Promise(r => setTimeout(r, REPLAY_FINALIZE_MS))
      setSentence(source, entry.sentence)
      if (entry.audio_b64) {
        setAudioUrl(source, base64ToAudioUrl(entry.audio_b64, entry.mime || 'audio/wav'))
      }
      markFinished(source)
    } catch (err) {
      console.error('Lỗi phát lại kết quả đã lưu:', err)
      setError(err?.message || 'Không phát được video.')
    } finally {
      setIsRunning(false)
      setPhase(null)
    }
  }

  const start = async () => {
    const video = videoRef.current
    if (!video || isRunning) return

    cancelRef.current = false
    setIsRunning(true)
    setError('')
    setProgress(0)
    setStats(null)

    // Clear previous run's result but keep translated sentence history.
    startNewRun(source)
    setActiveSource(source)
    setWsSink(source)

    // Record each word's timestamp so a later replay matches the live run.
    const timeline = []
    const unsubTimeline = useSignStore.subscribe((next, prev) => {
      const cur = next.sources[source].glosses
      const before = prev.sources[source].glosses
      for (let i = before.length; i < cur.length; i++) {
        timeline.push({ text: cur[i].text, at: Number((video.currentTime || 0).toFixed(3)) })
      }
    })

    try {
      const outcome = await streamVideoThroughWs(video, {
        onResults: handleResults,
        // Freshly recorded webm has duration = Infinity -> use the measured recording length.
        onProgress: (cur, total) => {
          const dur = Number.isFinite(total) && total > 0 ? total : item.duration
          setProgress(dur > 0 ? Math.min(cur / dur, 1) : 0)
        },
        onPhase: setPhase,
        onStats: setStats,
        isCancelled: () => cancelRef.current,
      })

      if (outcome === 'cancelled') {
        // Backend still holds glosses from the aborted run -> reset the connection for a fresh session.
        wsClient.resetSession()
      } else {
        setProgress(1)
        // Keep WS sink here until backend sends the final word + sentence + audio.
        setPhase('finalizing')
        await waitForUploadFinalize(source)
        await persistRun(timeline, video)
      }
    } catch (err) {
      console.error('Lỗi phát/dịch video:', err)
      setError(err?.message || 'Không phát được video qua MediaPipe.')
      wsClient.resetSession()
    } finally {
      unsubTimeline()
      setIsRunning(false)
      setPhase(null)
      setWsSink('realtime')
    }
  }

  /** Saves the just-finished run to cache; errors here don't affect the displayed result. */
  const persistRun = async (timeline, video) => {
    if (!item.cacheKey || timeline.length === 0) return
    const result = useSignStore.getState().sources[source]
    if (!result.sentence) return // no sentence means nothing worth caching
    const duration = Number.isFinite(video.duration) && video.duration > 0
      ? video.duration
      : item.duration || null
    const audioB64 = await blobUrlToBase64(result.audioUrl)
    const ok = await saveCachedResult(item.cacheKey, {
      glosses: timeline,
      sentence: result.sentence,
      duration,
      audioB64,
    })
    if (ok) {
      setCached({
        glosses: timeline,
        sentence: result.sentence,
        duration,
        audio_b64: audioB64,
        mime: 'audio/wav',
        source: 'live',
      })
    }
  }

  // Recorded clip: translate immediately, no extra button press needed.
  useEffect(() => {
    if (!autoStart || autoStartedRef.current) return
    if (cached) {
      autoStartedRef.current = true
      replayCached(cached)
      return
    }
    if (wsStatus !== 'connected') return
    autoStartedRef.current = true
    start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart, wsStatus, cached])

  return (
    <div className="flex-1 min-h-0 flex flex-col rounded-2xl border border-slate-200 dark:border-slate-900 bg-white/40 dark:bg-slate-950/25 overflow-hidden">
      <div className="relative flex-1 min-h-0 bg-slate-950 flex items-center justify-center">
        <video
          ref={videoRef}
          src={item.url}
          className="w-full h-full object-contain"
          muted
          playsInline
          preload="auto"
        />

        {showSkeleton && (
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none z-10"
          />
        )}

        {/* Video stays paused until MediaPipe finishes loading */}
        {phase === 'warmup' && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-slate-950/70 backdrop-blur-sm">
            <Loader2 size={28} className="animate-spin text-indigo-400" />
            <p className="text-xs font-semibold text-slate-200">Đang nạp mô hình AI...</p>
            <p className="text-[11px] text-slate-400">Lần đầu có thể mất một lúc để tải từ CDN.</p>
          </div>
        )}

        {phase === 'playing' && (
          <div className="absolute top-3 left-3 z-20 flex items-center gap-2 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-800 text-[10px] font-bold text-slate-200 uppercase tracking-wider">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span>Đang dịch</span>
          </div>
        )}

        {/* Below ~6.5fps the backend silently drops signs -> warn the user */}
        {phase === 'playing' && stats?.degraded && (
          <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-amber-500/90 text-white backdrop-blur-md px-3 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider">
            <AlertTriangle size={12} />
            <span>AI chậm — có thể sót từ</span>
          </div>
        )}

        <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-800/70 z-20">
          <div
            className="h-full bg-indigo-500 transition-[width] duration-150"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
      </div>

      <div className="shrink-0 flex items-center justify-between gap-3 px-4 py-2.5 border-t border-slate-200 dark:border-slate-900">
        <p className="text-[11px] text-slate-500 dark:text-slate-400 min-w-0 truncate">
          {error ? (
            <span className="text-rose-500">{error}</span>
          ) : !isRunning && noSign ? (
            <span className="text-amber-500">
              Không nhận được ký hiệu nào đủ tin cậy — thử ký rõ hơn, hạ tay giữa các từ.
            </span>
          ) : phase === 'warmup' ? (
            'Đang nạp mô hình AI...'
          ) : phase === 'finalizing' ? (
            'Đang ghép câu & tạo giọng nói...'
          ) : phase === 'playing' && stats ? (
            <>
              {`${Math.round(progress * 100)}% · AI `}
              <span className={stats.degraded ? 'text-amber-500 font-bold' : 'text-emerald-500 font-bold'}>
                {stats.fps.toFixed(1)} fps
              </span>
            </>
          ) : phase === 'playing' ? (
            `${Math.round(progress * 100)}% · đang dịch`
          ) : wsStatus === 'connected' ? (
            'Bấm để phát — ký hiệu dịch ngay ở panel bên phải'
          ) : (
            'Chưa kết nối được máy chủ dịch'
          )}
        </p>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={toggleSkeleton}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-semibold transition-all ${
              showSkeleton
                ? 'bg-emerald-500 border-emerald-400 text-white hover:bg-emerald-600'
                : 'bg-slate-100 dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-emerald-600 dark:text-emerald-400 hover:bg-slate-200 dark:hover:bg-slate-800'
            }`}
          >
            <Activity size={13} />
            <span>Khung xương</span>
          </button>

          {isRunning ? (
            <button
              onClick={stop}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-rose-500 hover:bg-rose-600 text-xs font-bold text-white shadow-md shadow-rose-500/20 transition-all"
            >
              <Square size={13} className="fill-white" />
              <span>Dừng</span>
            </button>
          ) : (
            <button
              onClick={() => (cached ? replayCached(cached) : start())}
              disabled={!cached && wsStatus !== 'connected'}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed text-xs font-bold text-white shadow-md shadow-indigo-500/20 transition-all"
            >
              {wsStatus === 'connecting' ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
              <span>Dịch trực tiếp</span>
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
