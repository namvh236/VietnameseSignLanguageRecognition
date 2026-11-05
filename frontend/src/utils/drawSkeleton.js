/**
 * Draws MediaPipe skeleton (no face mesh — too many segments per frame); canvas must match the video's size/object-fit to align correctly.
 */

export function drawSkeleton(canvas, results, width, height) {
  const ctx = canvas?.getContext('2d')
  if (!ctx || !width || !height) return

  // Resize only on change — setting canvas.width every frame reallocates its buffer.
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const { drawConnectors, drawLandmarks, POSE_CONNECTIONS, HAND_CONNECTIONS } = window
  if (!drawConnectors || !drawLandmarks) return

  if (results.poseLandmarks && POSE_CONNECTIONS) {
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: '#6366f1',
      lineWidth: 2,
    })
    drawLandmarks(ctx, results.poseLandmarks, {
      color: '#ffffff',
      fillColor: '#6366f1',
      lineWidth: 1.2,
      radius: 2.5,
    })
  }

  const hands = [
    [results.leftHandLandmarks, '#10b981'],
    [results.rightHandLandmarks, '#f59e0b'],
  ]
  for (const [landmarks, color] of hands) {
    if (!landmarks || !HAND_CONNECTIONS) continue
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color, lineWidth: 1.5 })
    drawLandmarks(ctx, landmarks, {
      color: '#ffffff',
      fillColor: color,
      lineWidth: 1,
      radius: 2,
    })
  }
}
