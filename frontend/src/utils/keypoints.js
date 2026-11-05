/**
 * SPOTER vector: 108 numbers; joint order = 12 body + 21 left hand + 21 right hand; neck (index 1) and missing joints = [0, 0]; coords in pixels.
 */

// MediaPipe Pose landmark indices in JointSelect body order (null = synthesized neck).
const POSE_MAP = [0, null, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15]

// MediaPipe Hand landmark indices in JointSelect order (used for both hands).
const HAND_MAP = [0, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 19, 20, 18, 17, 4, 3, 2, 1]

const KEYPOINTS_LEN = (POSE_MAP.length + HAND_MAP.length * 2) * 2 // 108

export function extractKeypoints(results, width, height) {
  const out = new Array(KEYPOINTS_LEN).fill(0)
  let o = 0

  const pose = results.poseLandmarks
  for (const idx of POSE_MAP) {
    const lm = idx !== null && pose ? pose[idx] : null
    if (lm) {
      out[o] = lm.x * width
      out[o + 1] = lm.y * height
    }
    o += 2
  }

  for (const hand of [results.leftHandLandmarks, results.rightHandLandmarks]) {
    for (const idx of HAND_MAP) {
      const lm = hand ? hand[idx] : null
      if (lm) {
        out[o] = lm.x * width
        out[o + 1] = lm.y * height
      }
      o += 2
    }
  }

  return out
}

// SL-GCN branch: 81 nums (27 joints x x,y,confidence) = 7 body + 10 left hand + 10 right hand.
// Confidence mirrors pose_format: body = MediaPipe visibility, hand = 1.0 if detected else 0,
// missing joint = [0,0,0]. Must stay bit-exact with backend normalize_distribution.

// SL-GCN body landmarks: nose, L/R shoulder, L/R elbow, L/R wrist.
const SLGCN_POSE_MAP = [0, 11, 12, 13, 14, 15, 16]

// SL-GCN hand landmarks (both hands): wrist, thumb/index/middle/ring/pinky MCP + tip.
const SLGCN_HAND_MAP = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

const SLGCN_KEYPOINTS_LEN = (SLGCN_POSE_MAP.length + SLGCN_HAND_MAP.length * 2) * 3 // 81

export function extractSlgcnKeypoints(results, width, height) {
  const out = new Array(SLGCN_KEYPOINTS_LEN).fill(0)
  let o = 0

  const pose = results.poseLandmarks
  for (const idx of SLGCN_POSE_MAP) {
    const lm = pose ? pose[idx] : null
    if (lm) {
      out[o] = lm.x * width
      out[o + 1] = lm.y * height
      out[o + 2] = lm.visibility ?? 0 // body confidence = visibility
    }
    o += 3
  }

  // Left hand then right hand, matching pose_format's LEFT/RIGHT_HAND_LANDMARKS order.
  for (const hand of [results.leftHandLandmarks, results.rightHandLandmarks]) {
    for (const idx of SLGCN_HAND_MAP) {
      const lm = hand ? hand[idx] : null
      if (lm) {
        out[o] = lm.x * width
        out[o + 1] = lm.y * height
        out[o + 2] = 1.0 // hand confidence = 1.0 when detected
      }
      o += 3
    }
  }

  return out
}
