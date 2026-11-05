/**
 * Cache key = SHA-256 of file bytes, matching Python's hashlib.sha256 (shared with the prebake script); falls back to a weak name+size+mtime hash without crypto.subtle (e.g. LAN http).
 */

/** Hashes a string to a 64-char hex string (four-round FNV-1a with different seeds). */
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
