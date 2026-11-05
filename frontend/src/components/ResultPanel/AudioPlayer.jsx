import { useRef, useEffect, useState } from 'react'
import { Volume2, RotateCcw, Loader2, VolumeX } from 'lucide-react'
import useSignStore from '../../store/useSignStore'

export default function AudioPlayer({ audioUrl }) {
  const audioRef = useRef(new Audio())
  const voiceSpeed = useSignStore(s => s.voiceSpeed)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [playError, setPlayError] = useState(false)

  useEffect(() => {
    const audio = audioRef.current
    setIsLoading(true)
    setPlayError(false)
    audio.src = audioUrl

    const handleCanPlay = () => {
      setIsLoading(false)
      audio.play()
        .then(() => {
          setIsPlaying(true)
          setPlayError(false)
        })
        .catch((err) => {
          console.log('Autoplay prevented or failed:', err)
          setIsPlaying(false)
          // Chrome/Safari prevent autoplay without interaction
        })
    }

    const handleEnded = () => {
      setIsPlaying(false)
    }

    const handleError = () => {
      setIsLoading(false)
      setPlayError(true)
    }

    audio.addEventListener('canplaythrough', handleCanPlay)
    audio.addEventListener('ended', handleEnded)
    audio.addEventListener('error', handleError)

    return () => {
      audio.pause()
      audio.src = ''
      audio.removeEventListener('canplaythrough', handleCanPlay)
      audio.removeEventListener('ended', handleEnded)
      audio.removeEventListener('error', handleError)
    }
  }, [audioUrl])

  // Apply the chosen speed (x button in ResultPanel); changing src resets playbackRate
  // to defaultPlaybackRate, so set both here and rerun when audio changes.
  useEffect(() => {
    const audio = audioRef.current
    audio.defaultPlaybackRate = voiceSpeed
    audio.playbackRate = voiceSpeed
  }, [voiceSpeed, audioUrl])

  const replay = () => {
    const audio = audioRef.current
    audio.currentTime = 0
    audio.play()
      .then(() => {
        setIsPlaying(true)
        setPlayError(false)
      })
      .catch((err) => {
        console.error('Audio playback failed:', err)
        setPlayError(true)
      })
  }

  return (
    <div className="flex items-center gap-4 p-4 rounded-2xl bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-100 dark:border-indigo-900/30 backdrop-blur-sm transition-all duration-300">
      <div className={`p-2.5 rounded-xl transition-all duration-300 ${isPlaying ? 'bg-indigo-500/20 text-indigo-500 dark:text-indigo-400' : 'bg-slate-200 dark:bg-slate-900 text-slate-500 dark:text-slate-400'}`}>
        {playError ? <VolumeX size={18} className="text-rose-500 dark:text-rose-400" /> : <Volume2 size={18} />}
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">Đọc phát âm (TTS)</p>
        <p className="text-xs text-slate-500 dark:text-slate-400 truncate mt-0.5">
          {isLoading ? 'Đang chuẩn bị âm thanh...' : isPlaying ? 'Đang phát âm thanh...' : playError ? 'Lỗi phát âm thanh' : 'Sẵn sàng để phát lại'}
        </p>
      </div>

      {/* Decorative simple voice waveforms when playing */}
      {isPlaying && (
        <div className="flex items-end gap-0.5 h-4 px-2">
          <span className="w-0.5 h-2 bg-indigo-500 dark:bg-indigo-400 rounded-full animate-[bounce_0.8s_infinite]" />
          <span className="w-0.5 h-4 bg-indigo-500 dark:bg-indigo-400 rounded-full animate-[bounce_0.6s_infinite_0.1s]" />
          <span className="w-0.5 h-3 bg-indigo-500 dark:bg-indigo-400 rounded-full animate-[bounce_0.7s_infinite_0.2s]" />
          <span className="w-0.5 h-1 bg-indigo-500 dark:bg-indigo-400 rounded-full animate-[bounce_0.5s_infinite_0.3s]" />
        </div>
      )}

      {isLoading ? (
        <div className="p-2">
          <Loader2 size={16} className="animate-spin text-indigo-500 dark:text-indigo-400" />
        </div>
      ) : (
        <button
          onClick={replay}
          title="Phát lại âm thanh"
          className="p-2 rounded-xl bg-slate-200 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:border-slate-400 dark:hover:border-slate-700 transition-all duration-200"
        >
          <RotateCcw size={16} />
        </button>
      )}
    </div>
  )
}
