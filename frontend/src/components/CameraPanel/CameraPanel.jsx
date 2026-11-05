import { useRef, useCallback } from 'react'
import useCamera from '../../hooks/useCamera'
import useSignStore from '../../store/useSignStore'
import VideoDisplay from './VideoDisplay'
import ManualPanel from '../ManualPanel/ManualPanel'
import wsClient from '../../services/ws'

const IDLE_FINALIZE_MS = 2500 // idle 2.5s -> auto-finalize sentence + speak it

export default function CameraPanel({ activeTab }) {
  const videoRef = useRef(null)
  const isCameraOn = useSignStore(s => s.isCameraOn)
  const { error } = useCamera(videoRef, isCameraOn)

  // MediaPipe Holistic (in VideoDisplay) calls back every frame with extracted keypoints
  // (SPOTER 108 values + SL-GCN 81 values for late-fusion).
  const handleKeypoints = useCallback((keypoints, slgcnKeypoints) => {
    wsClient.sendKeypoints(keypoints, slgcnKeypoints)
    if (wsClient.shouldFinalize(IDLE_FINALIZE_MS)) {
      wsClient.sendKeypoints(keypoints, slgcnKeypoints, true)
    }
  }, [])

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Realtime video display or Manual Upload panel */}
      {activeTab === 'manual' ? (
        <ManualPanel />
      ) : (
        <VideoDisplay videoRef={videoRef} error={error} onKeypoints={handleKeypoints} />
      )}
    </div>
  )
}


