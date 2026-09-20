/**
 * MediaPipe Holistic dùng chung cho camera realtime và trình phát video upload: model chỉ
 * tải từ CDN một lần, không có hai đồ thị chạy song song tranh GPU.
 *
 * Chỉ MỘT nơi tiêu thụ kết quả tại một thời điểm — setResultsHandler() ghi đè handler cũ.
 */

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic'

let holistic = null
let initPromise = null
let resultsHandler = null

/** Chờ script MediaPipe từ CDN tải xong (index.html nạp bằng thẻ <script>). */
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

/** Lấy (và khởi tạo nếu cần) instance Holistic dùng chung. */
export function getHolistic() {
  if (holistic) return Promise.resolve(holistic)
  if (!initPromise) {
    initPromise = (async () => {
      await waitForScript()
      const instance = new window.Holistic({ locateFile: (file) => `${CDN}/${file}` })
      // Giữ nguyên tham số của luồng camera để keypoint cùng phân bố với dữ liệu mà
      // ngưỡng nhận dạng phía backend đã căn theo.
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
      initPromise = null // cho phép thử lại lần sau
      throw err
    })
  }
  return initPromise
}

/** Đăng ký nơi nhận kết quả mỗi frame. Truyền null khi rời màn hình để nhả quyền. */
export function setResultsHandler(fn) {
  resultsHandler = fn
}

/**
 * Đổi độ phức tạp model khi đang chạy (0 = Lite, nhanh hơn nhưng bắt bàn tay kém hơn).
 * Instance dùng chung nên nơi gọi phải trả lại mức cũ. setOptions() nạp lại đồ thị model
 * (~1 giây) — chỉ gọi lúc bắt đầu/kết thúc, đừng gọi giữa vòng lặp frame.
 */
export function setModelComplexity(complexity) {
  if (!holistic) return
  holistic.setOptions({ modelComplexity: complexity })
}

export function isHolisticReady() {
  return holistic !== null
}

/**
 * Vứt instance để lần getHolistic() sau tạo lại từ đầu. Cần khi MediaPipe mất ngữ cảnh
 * WebGL — instance cũ hỏng vĩnh viễn, mọi send() sau đó đều ném lỗi.
 */
export function disposeHolistic() {
  const instance = holistic
  holistic = null
  initPromise = null
  resultsHandler = null
  try {
    instance?.close?.()
  } catch {
    // Instance đã hỏng thì close() cũng có thể ném — không sao, ta bỏ nó đi rồi.
  }
}
