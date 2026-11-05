import { useState, useEffect } from 'react'

export default function useCamera(videoRef, isCameraOn = true) {
  const [error, setError] = useState(null)

  useEffect(() => {
    let stream = null
    // getUserMedia is async: if the effect is cleaned up before it resolves, stop the stream
    // immediately on arrival so the camera doesn't stay on.
    let cancelled = false

    const start = async () => {
      if (!isCameraOn) {
        return
      }

      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error('Trình duyệt không hỗ trợ truy cập camera.')
        }

        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: false,
        })

        if (cancelled) {
          stream.getTracks().forEach(t => t.stop())
          return
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          try {
            await videoRef.current.play()
          } catch (playErr) {
            if (playErr.name !== 'AbortError') {
              throw playErr
            }
          }
        }
      } catch (err) {
        if (cancelled) return
        console.error('Camera capture error:', err)
        setError(err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError'
          ? 'Vui lòng cấp quyền truy cập camera để tiếp tục.'
          : 'Không thể mở camera: ' + err.message
        )
      }
    }

    start()

    return () => {
      cancelled = true
      if (stream) {
        stream.getTracks().forEach(t => t.stop())
      }
    }
  }, [videoRef, isCameraOn])

  return { error }
}
