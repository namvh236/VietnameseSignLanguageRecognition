/**
 * Khoá cache của một file video = SHA-256 nội dung file (khớp với hashlib.sha256 phía Python,
 * nên script prebake và web dùng chung khoá). Đổi tên file không đổi khoá.
 *
 * crypto.subtle chỉ có trong ngữ cảnh bảo mật (https hoặc localhost). Demo chạy qua IP LAN
 * bằng http thì không có -> lui về khoá yếu hơn từ (tên + kích thước + thời điểm sửa). Khoá
 * yếu này KHÔNG khớp với script prebake, nhưng vẫn dùng lại được kết quả đã chạy trên máy đó.
 */

/** Băm chuỗi -> 64 ký tự hex (FNV-1a bốn vòng với hằng số khác nhau). */
function weakHex(text) {
  const seeds = [0x811c9dc5, 0x01000193, 0x9e3779b9, 0x7feb352d]
  return seeds
    .map((seed) => {
      let h = seed >>> 0
      for (let i = 0; i < text.length; i++) {
        h ^= text.charCodeAt(i)
        h = Math.imul(h, 0x01000193) >>> 0
      }
      return h.toString(16).padStart(8, '0')
    })
    .join('')
    .padEnd(32, '0')
}

export async function videoCacheKey(file) {
  if (globalThis.crypto?.subtle) {
    try {
      const buf = await file.arrayBuffer()
      const digest = await crypto.subtle.digest('SHA-256', buf)
      return Array.from(new Uint8Array(digest))
        .map(b => b.toString(16).padStart(2, '0'))
        .join('')
    } catch (err) {
      console.warn('Không băm được nội dung file, dùng khoá dự phòng:', err)
    }
  }
  return weakHex(`${file.name}|${file.size}|${file.lastModified}`)
}
