/**
 * Cache kết quả dịch theo file video (lưới an toàn cho demo): video đã xử lý một lần thì lần
 * sau phát lại ngay, không chạy lại MediaPipe/ONNX/LLM/TTS.
 *
 * Backend: GET/POST /api/cache/<key> (xem backend/serving/routers/cache.py).
 */

const BASE_URL = import.meta.env.VITE_API_URL

/** Kết quả đã lưu, hoặc null nếu video này chưa từng được xử lý (hoặc backend không có cache). */
export async function fetchCachedResult(key) {
  if (!key || !BASE_URL) return null
  try {
    const res = await fetch(`${BASE_URL}/api/cache/${key}`)
    if (res.status === 404) return null
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json() // { glosses: [{text, at}], sentence, duration, audio_b64, mime, source }
  } catch (err) {
    console.warn('Không đọc được cache kết quả:', err)
    return null
  }
}

/** Lưu kết quả một lần chạy thật. Lỗi ở đây không được ảnh hưởng tới lượt dịch vừa xong. */
export async function saveCachedResult(key, { glosses, sentence, duration, audioB64 }) {
  if (!key || !BASE_URL || !glosses?.length) return false
  try {
    const res = await fetch(`${BASE_URL}/api/cache/${key}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        glosses,
        sentence,
        duration: duration ?? null,
        audio_b64: audioB64 ?? null,
      }),
    })
    return res.ok
  } catch (err) {
    console.warn('Không lưu được cache kết quả:', err)
    return false
  }
}

/** Blob URL (WAV) -> base64 để gửi kèm khi lưu cache. */
export async function blobUrlToBase64(url) {
  if (!url) return null
  try {
    const blob = await (await fetch(url)).blob()
    const buf = new Uint8Array(await blob.arrayBuffer())
    let binary = ''
    const CHUNK = 0x8000 // String.fromCharCode(...) tràn stack nếu đẩy cả file một lần
    for (let i = 0; i < buf.length; i += CHUNK) {
      binary += String.fromCharCode.apply(null, buf.subarray(i, i + CHUNK))
    }
    return btoa(binary)
  } catch (err) {
    console.warn('Không đọc được audio để lưu cache:', err)
    return null
  }
}
