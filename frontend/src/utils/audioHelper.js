const _urls = new Set()

/** Converts a base64 audio string (from WS audio_b64) to a blob URL. */
export const base64ToAudioUrl = (base64, mimeType = 'audio/wav') => {
  const binaryString = atob(base64)
  const len = binaryString.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  const blob = new Blob([bytes], { type: mimeType })
  const url = URL.createObjectURL(blob)
  _urls.add(url)
  return url
}

/** Creates a blob URL from a binary WAV frame sent by the backend over WebSocket. */
export const blobToAudioUrl = (blob) => {
  const url = URL.createObjectURL(blob)
  _urls.add(url)
  return url
}

export const revokeAudioUrl = (url) => {
  if (url?.startsWith('blob:')) {
    URL.revokeObjectURL(url)
    _urls.delete(url)
  }
}

window.addEventListener('beforeunload', () => {
  _urls.forEach(url => URL.revokeObjectURL(url))
})
