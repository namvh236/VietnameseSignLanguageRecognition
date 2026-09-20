/**
 * Phát một file video qua đường realtime: MediaPipe Holistic (trình duyệt) -> keypoint ->
 * WebSocket -> backend cắt ký hiệu & dịch.
 *
 * Hai vòng chạy độc lập:
 * - `pump`: MediaPipe theo nhịp riêng, xong frame nào làm ngay frame kế tiếp.
 * - `tick`: gửi keypoint đúng nhịp fps của video, lặp lại keypoint gần nhất cho frame
 *   MediaPipe chưa kịp tính (hold-last). Backend đếm ngưỡng cắt ký hiệu theo SỐ FRAME nên
 *   số frame gửi lên phải khớp.
 *
 * Phải hâm nóng MediaPipe trước khi phát: model (~10MB từ CDN) chỉ tải ở lần send() đầu tiên,
 * phát ngay thì cả clip trôi qua trước khi có keypoint nào.
 */

import { extractKeypoints, extractSlgcnKeypoints } from '../utils/keypoints'
import { disposeHolistic, getHolistic, setResultsHandler } from './holistic'
import wsClient from './ws'

/**
 * MediaPipe ném ĐỒNG BỘ khi không tạo được ngữ cảnh WebGL, nên `send().catch()` không bắt
 * được — ngoại lệ thoát ra và giết cả vòng lặp. Bọc lại để mọi lỗi đi chung một đường.
 */
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

/** Thiếu WebGL thì MediaPipe chết bên trong với đủ kiểu lỗi khó hiểu — gom về một thông báo. */
function describeMediaPipeError(err) {
  const msg = String(err?.message || err || '')
  if (/webgl|gl context|graphics|changeOptions|Cannot read properties|wasm|abort/i.test(msg)) {
    return `${WEBGL_HINT} (chi tiết: ${msg})`
  }
  return msg || 'Không phát được video qua MediaPipe.'
}

const DEFAULT_FPS = 25
const EMIT_INTERVAL_MS = 33
const FLOOR_SAMPLE_FPS = 6.5 // dưới mức này ký hiệu bị nuốt im lặng -> cảnh báo
const WARMUP_TIMEOUT_MS = 60000
const WARMUP_RETRY_MS = 250
const WARMUP_RESULTS = 3 // vài frame đầu của MediaPipe luôn chậm; chờ nó ổn định rồi mới phát
const UI_UPDATE_MS = 400 // hạ nhịp render React để không tranh luồng chính với MediaPipe
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

/**
 * @param {HTMLVideoElement} video
 * @param {object} opts
 *   onResults(results)      - mỗi khi MediaPipe trả kết quả (vẽ khung xương)
 *   onProgress(cur, total)  - tiến độ (giây), đã hạ nhịp
 *   onPhase(phase)          - 'warmup' | 'playing'
 *   onStats({fps, degraded})
 *   isCancelled()
 * @returns {Promise<'completed'|'cancelled'>}
 */
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

  let latest = null // keypoint mới nhất: [spoter108, slgcn81]
  let firstResult = null // resolve của lần hâm nóng
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

  // Mất ngữ cảnh WebGL là instance hỏng vĩnh viễn — thử dựng lại một lần.
  // disposeHolistic() xoá cả handler nên phải đăng ký lại.
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
    // ---- Giai đoạn 1: hâm nóng (video đứng yên) ----
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
      // Bắn liên tục cho tới khi model tải xong và trả về đủ kết quả.
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
            // Vài lần đầu có thể chỉ là frame chưa sẵn sàng.
            if (failures === 3) {
              try {
                await recoverHolistic()
                return
              } catch {
                // Dựng lại cũng không xong -> rơi xuống nhánh báo lỗi bên dưới.
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

    // ---- Giai đoạn 2: phát và dịch ----
    onPhase?.('playing')
    return await new Promise((resolve, reject) => {
      let done = false
      // Dùng 25 fps ngay, không chờ đo: nhịp gửi phải chạy từ frame đầu tiên.
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
        // end_sign: backend flush ký hiệu đang dở rồi ghép câu + đọc.
        if (latest) wsClient.sendKeypoints(latest[0], latest[1], true)
        finish('completed')
      }

      /** Gửi bù cho tới đúng số frame video đã trôi qua (lặp lại keypoint gần nhất). */
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
              // Mất ngữ cảnh WebGL giữa chừng -> dựng lại instance rồi chạy tiếp.
              try {
                if (await recoverHolistic()) return
              } catch {
                // Không dựng lại được -> báo lỗi ở lần thứ 6.
              }
            }
            if (sendFailures >= 6) {
              fail(new Error(describeMediaPipeError(err)))
            }
          })
          .finally(() => {
            if (done || cancelled()) return
            pumpAliveAt = performance.now()
            // Nhường luồng chính một nhịp cho React và vòng gửi keypoint.
            setTimeout(pump, performance.now() - t0 < 5 ? 15 : 0)
          })
      }

      // Dùng setInterval chứ không requestVideoFrameCallback: rVFC bị bỏ qua khi luồng chính
      // bận. Mốc thời gian lấy từ video.currentTime nên timer trễ vẫn gửi bù đúng số frame.
      const tick = () => {
        if (done) return
        if (cancelled()) return finish('cancelled')

        catchUp(video.currentTime)

        const now = performance.now()
        if (pumpAliveAt && now - pumpAliveAt > PUMP_WATCHDOG_MS) {
          pumpAliveAt = now
          pump() // vòng MediaPipe kẹt -> khởi động lại
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

      /**
       * Đo fps song song với MediaPipe (hoãn MediaPipe để đo sẽ tạo vùng chết đầu video và
       * làm mất ký hiệu bắt đầu sớm). Lấy khoảng cách NHỎ NHẤT chứ không phải trung vị: frame
       * bị bỏ chỉ làm khoảng cách lớn hơn. Số đo vô lý thì giữ mặc định 25 fps.
       */
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
            emitted = Math.floor(video.currentTime * fps) // đồng bộ lại mốc đếm
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
