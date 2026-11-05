/**
 * Shared MediaPipe Holistic instance for camera + video (avoids loading it twice or fighting over the GPU context); setResultsHandler() replaces the previous handler, so only one consumer runs at a time.
 */

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic'

let holistic = null
let initPromise = null
let resultsHandler = null

/** Waits for the MediaPipe script tag (loaded in index.html) to finish loading. */
function waitForScript(timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    if (window.Holistic) return resolve()
    const startedAt = Date.now()
    const tick = () => {
      if (window.Holistic) return resolve()
      if (Date.now() - startedAt >= timeoutMs) {
        return reject(new Error('Không tải được MediaPipe Holistic từ CDN.'))
      }
      setTimeout(tick, 200)
    }
    tick()
  })
}

export function getHolistic() {
  if (holistic) return Promise.resolve(holistic)
  if (!initPromise) {
    initPromise = (async () => {
      await waitForScript()
      const instance = new window.Holistic({ locateFile: (file) => `${CDN}/${file}` })
      // Keep params identical to the camera stream: backend recognition thresholds are
      // calibrated to this exact keypoint distribution.
      instance.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.4,
        minTrackingConfidence: 0.4,
      })
      instance.onResults((results) => resultsHandler?.(results))
      holistic = instance
      return instance
    })().catch((err) => {
      initPromise = null // allow a retry next time
      throw err
    })
  }
  return initPromise
}

/** Registers the per-frame results handler; pass null on unmount to release it. */
export function setResultsHandler(fn) {
  resultsHandler = fn
}

/**
 * Runtime model-complexity change (0 = Lite: faster, worse hand tracking); reloads the model graph (~1s) — call only at start/end, never mid-frame-loop; shared instance, so restore the previous value after.
 */
export function setModelComplexity(complexity) {
  if (!holistic) return
  holistic.setOptions({ modelComplexity: complexity })
}

export function isHolisticReady() {
  return holistic !== null
}

/**
 * Drops the instance so the next getHolistic() recreates it; needed when WebGL context loss permanently breaks the old instance (later send() calls all throw).
 */
export function disposeHolistic() {
  const instance = holistic
  holistic = null
  initPromise = null
  resultsHandler = null
  try {
    instance?.close?.()
  } catch {
    // instance may already be broken; close() can throw too — safe to ignore, we're discarding it
  }
}
