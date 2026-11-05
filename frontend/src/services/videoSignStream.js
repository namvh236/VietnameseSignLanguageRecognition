/** Streams a video file through the realtime pipeline: MediaPipe -> keypoints -> WebSocket. Two independent loops — `pump` runs MediaPipe at its own pace, `tick` emits keypoints at the video's fps, holding the last frame when MediaPipe lags (backend cuts signs by frame count, so counts must match). Must warm up MediaPipe before playing: the ~10MB CDN model only loads on the first send(). */

import { extractKeypoints, extractSlgcnKeypoints } from '../utils/keypoints'
import { disposeHolistic, getHolistic, setResultsHandler } from './holistic'
import wsClient from './ws'

/** MediaPipe throws synchronously when it can't create a WebGL context, so send().catch() won't catch it and the exception kills the loop; wrap it so every error goes through one path. */
function safeSend(holistic, video) {
  try {
    const p = holistic.send({ image: video })
    return p && typeof p.then === 'function' ? p : Promise.resolve()
  } catch (err) {
    return Promise.reject(err)
  }
}

const WEBGL_HINT =
  'MediaPipe không dùng được WebGL/GPU của trình duyệt. Thường do quá nhiều tab đang chiếm ' +
  'ngữ cảnh WebGL hoặc tiến trình GPU vừa khởi động lại — đóng bớt tab rồi tải lại trang.'

/** MediaPipe fails internally with cryptic errors when WebGL is unavailable — map them to one clear message. */
function describeMediaPipeError(err) {
  const msg = String(err?.message || err || '')
  if (/webgl|gl context|graphics|changeOptions|Cannot read properties|wasm|abort/i.test(msg)) {
    return `${WEBGL_HINT} (chi tiết: ${msg})`
  }
  return msg || 'Không phát được video qua MediaPipe.'
}

const DEFAULT_FPS = 25
const EMIT_INTERVAL_MS = 33
const FLOOR_SAMPLE_FPS = 6.5 // below this, signs get silently dropped -> warn
const WARMUP_TIMEOUT_MS = 60000
const WARMUP_RETRY_MS = 250
const WARMUP_RESULTS = 3 // the first few MediaPipe frames are always slow; wait for it to stabilize
const UI_UPDATE_MS = 400 // throttle React renders so they don't compete with MediaPipe on the main thread
const PUMP_WATCHDOG_MS = 3000
const STATS_WINDOW_MS = 1000

export function onceEvent(el, event, timeoutMs = 5000) {
  return new Promise((resolve) => {
    let settled = false
    const finish = () => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      el.removeEventListener(event, finish)
      resolve()
    }
    const timer = setTimeout(finish, timeoutMs)
    el.addEventListener(event, finish, { once: true })
  })
}

/** Streams `video` to the backend over WS, calling onResults/onProgress/onPhase/onStats as it runs; resolves to 'completed' or 'cancelled'. */
export async function streamVideoThroughWs(
  video,
  { onResults, onProgress, onPhase, onStats, isCancelled } = {}
) {
  const cancelled = () => (isCancelled ? isCancelled() : false)
  let holistic
  try {
    holistic = await getHolistic()
  } catch (err) {
    disposeHolistic()
    throw new Error(describeMediaPipeError(err))
  }

  let latest = null // latest keypoint pair: [spoter108, slgcn81]
  let firstResult = null // resolves the warmup promise
  let resultCount = 0

  const handleFrameResults = (results) => {
    resultCount++
    const w = video.videoWidth
    const h = video.videoHeight
    if (w && h) {
      latest = [extractKeypoints(results, w, h), extractSlgcnKeypoints(results, w, h)]
    }
    firstResult?.()
    onResults?.(results)
  }
  setResultsHandler(handleFrameResults)

  // WebGL context loss permanently breaks the instance — try rebuilding it once.
  // disposeHolistic() also clears the handler, so re-register it.
  let recoveryUsed = false
  const recoverHolistic = async () => {
    if (recoveryUsed) return false
    recoveryUsed = true
    disposeHolistic()
    holistic = await getHolistic()
    setResultsHandler(handleFrameResults)
    return true
  }

  try {
    onPhase?.('warmup')
    if (video.readyState < 2) await onceEvent(video, 'loadeddata', 10000)
    video.pause()
    video.currentTime = 0
    await onceEvent(video, 'seeked', 3000)

    let warm = false
    await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('MediaPipe không phản hồi sau 60 giây (không tải được model?)')),
        WARMUP_TIMEOUT_MS
      )
      let warmCount = 0
      firstResult = () => {
        if (warm) return
        if (++warmCount < WARMUP_RESULTS) return
        warm = true
        clearTimeout(timer)
        resolve()
      }
      // keep sending until the model finishes loading and returns enough results
      let fatal = false
      let failures = 0
      const kick = () => {
        if (warm || cancelled()) return void (warm = true, clearTimeout(timer), resolve())
        safeSend(holistic, video)
          .then(() => {
            failures = 0
          })
          .catch(async (err) => {
            failures++
            // first few failures may just be frame-not-ready
            if (failures === 3) {
              try {
                await recoverHolistic()
                return
              } catch {
                // rebuild failed too -> fall through to the error branch below
              }
            }
            if (failures >= 6) {
              fatal = true
              clearTimeout(timer)
              reject(new Error(describeMediaPipeError(err)))
            }
          })
          .finally(() => {
            if (!warm && !fatal) setTimeout(kick, WARMUP_RETRY_MS)
          })
      }
      kick()
    })
    if (cancelled()) return 'cancelled'

    onPhase?.('playing')
    return await new Promise((resolve, reject) => {
      let done = false
      // use 25fps immediately (don't wait to measure) — emission must start from frame one
      let fps = DEFAULT_FPS
      let fpsMeasured = false
      let emitted = 0
      let lastUi = 0
      let pumpAliveAt = 0
      let emitTimer = null
      let windowStart = performance.now()
      let windowBase = resultCount
      let mediapipeFps = 0
      const mediaTimes = []

      const cleanup = () => {
        if (emitTimer) clearInterval(emitTimer)
        video.removeEventListener('ended', onEnded)
        video.pause()
      }
      const finish = (outcome) => {
        if (done) return
        done = true
        cleanup()
        resolve(outcome)
      }
      const fail = (err) => {
        if (done) return
        done = true
        cleanup()
        reject(err)
      }

      function onEnded() {
        if (done || cancelled()) return
        // end_sign: backend flushes the pending sign, builds the sentence, and speaks it
        if (latest) wsClient.sendKeypoints(latest[0], latest[1], true)
        finish('completed')
      }

      /** Catches up to the video's current frame count, repeating the latest keypoint. */
      function catchUp(mediaTime) {
        const target = Math.floor(mediaTime * fps)
        if (!latest) {
          emitted = target
          return
        }
        for (let i = emitted; i < target; i++) {
          wsClient.sendKeypoints(latest[0], latest[1])
        }
        if (target > emitted) emitted = target
      }

      let sendFailures = 0
      const pump = () => {
        if (done || cancelled()) return
        if (video.readyState < 2) return void setTimeout(pump, 30)
        const t0 = performance.now()
        pumpAliveAt = t0
        safeSend(holistic, video)
          .then(() => {
            sendFailures = 0
          })
          .catch(async (err) => {
            sendFailures++
            if (sendFailures === 3) {
              // WebGL context lost mid-stream -> rebuild the instance and continue
              try {
                if (await recoverHolistic()) return
              } catch {
                // rebuild failed -> error reported at the 6th failure
              }
            }
            if (sendFailures >= 6) {
              fail(new Error(describeMediaPipeError(err)))
            }
          })
          .finally(() => {
            if (done || cancelled()) return
            pumpAliveAt = performance.now()
            // yield the main thread briefly for React and the keypoint-sending loop
            setTimeout(pump, performance.now() - t0 < 5 ? 15 : 0)
          })
      }

      // setInterval, not requestVideoFrameCallback: rVFC gets skipped when the main thread is
      // busy. Timing uses video.currentTime, so a delayed timer still catches up correctly.
      const tick = () => {
        if (done) return
        if (cancelled()) return finish('cancelled')

        catchUp(video.currentTime)

        const now = performance.now()
        if (pumpAliveAt && now - pumpAliveAt > PUMP_WATCHDOG_MS) {
          pumpAliveAt = now
          pump() // MediaPipe loop stuck -> restart it
        }
        if (now - windowStart >= STATS_WINDOW_MS) {
          mediapipeFps = ((resultCount - windowBase) * 1000) / (now - windowStart)
          windowStart = now
          windowBase = resultCount
        }
        if (now - lastUi >= UI_UPDATE_MS) {
          lastUi = now
          onProgress?.(video.currentTime, video.duration)
          onStats?.({ fps: mediapipeFps, degraded: mediapipeFps < FLOOR_SAMPLE_FPS })
        }
      }

      /** Measures fps alongside MediaPipe (pausing to measure would create a dead zone and miss early signs); uses the smallest frame gap, not the median, since dropped frames only inflate gaps — falls back to 25fps if unreasonable. */
      const measureFps = () => {
        if (done || typeof video.requestVideoFrameCallback !== 'function') return
        const deadline = performance.now() + 2500
        const onFrame = (_now, metadata) => {
          if (done || fpsMeasured) return
          mediaTimes.push(metadata.mediaTime)
          if (mediaTimes.length < 12 && performance.now() < deadline) {
            return void video.requestVideoFrameCallback(onFrame)
          }
          let smallest = Infinity
          for (let i = 1; i < mediaTimes.length; i++) {
            const d = mediaTimes[i] - mediaTimes[i - 1]
            if (d > 0.001 && d < smallest) smallest = d
          }
          const measured = 1 / smallest
          if (measured >= 15 && measured <= 61) {
            fps = measured
            emitted = Math.floor(video.currentTime * fps) // resync the frame counter
          }
          fpsMeasured = true
        }
        video.requestVideoFrameCallback(onFrame)
      }

      video.addEventListener('ended', onEnded)
      video.playbackRate = 1
      video
        .play()
        .then(() => {
          pump()
          emitTimer = setInterval(tick, EMIT_INTERVAL_MS)
          measureFps()
        })
        .catch(fail)
    })
  } finally {
    setResultsHandler(null)
    firstResult = null
  }
}
