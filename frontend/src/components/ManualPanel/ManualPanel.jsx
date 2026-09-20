import { useState, useRef, useEffect } from 'react'
import { UploadCloud, FileVideo, RefreshCw, Video, Clapperboard } from 'lucide-react'
import useSignStore from '../../store/useSignStore'
import { videoCacheKey } from '../../utils/videoHash'
import UploadPlayer from './UploadPlayer'
import VideoRecorder from './VideoRecorder'

/**
 * Tab "Nhận dạng thủ công", hai chế độ dùng CÙNG một đường dịch (phát video -> MediaPipe ->
 * keypoint -> WebSocket), chỉ khác nguồn video:
 * - 'upload': tải một tệp lên  -> kết quả vào nguồn 'upload'
 * - 'record': tự quay bằng webcam -> kết quả vào nguồn 'record'
 *
 * Mỗi chế độ chỉ giữ một clip — nạp clip khác là thay luôn, chỉ lịch sử câu đã dịch được giữ
 * lại (xem startNewRun trong store).
 */
export default function ManualPanel() {
  const [mode, setMode] = useState('upload') // 'upload' | 'record'
  const [video, setVideo] = useState(null) // clip tải lên: { id, file, name, url }
  const [recorded, setRecorded] = useState(null) // clip tự quay: thêm { duration }
  const fileInputRef = useRef(null)
  const setActiveSource = useSignStore(s => s.setActiveSource)

  // Chế độ đang mở chính là nguồn kết quả hiển thị ở ResultPanel.
  useEffect(() => {
    setActiveSource(mode)
  }, [mode, setActiveSource])

  // Thu hồi blob URL khi rời tab, tránh rò rỉ bộ nhớ.
  useEffect(() => {
    return () => {
      if (video?.url) URL.revokeObjectURL(video.url)
      if (recorded?.url) URL.revokeObjectURL(recorded.url)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const acceptFile = async (file) => {
    if (!file || !file.type.startsWith('video/')) return
    const id = `upload-${Date.now()}`
    setVideo(prev => {
      if (prev?.url) URL.revokeObjectURL(prev.url) // chỉ giữ một video tại một thời điểm
      return {
        id,
        file,
        name: file.name,
        url: URL.createObjectURL(file),
      }
    })
    // Băm nội dung file để tra cache kết quả (video đã xử lý trước thì phát lại ngay).
    // Chạy sau khi đã hiện video: file lớn mất một nhịp, không nên chặn UI.
    try {
      const cacheKey = await videoCacheKey(file)
      setVideo(prev => (prev && prev.id === id ? { ...prev, cacheKey } : prev))
    } catch (err) {
      console.warn('Không tạo được khoá cache cho video:', err)
    }
  }

  const handleFileChange = (e) => {
    acceptFile(e.target.files?.[0])
    e.target.value = '' // chọn lại đúng tệp cũ vẫn kích hoạt onChange
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (mode !== 'upload') return
    acceptFile(e.dataTransfer.files?.[0])
  }

  const acceptRecording = (item) => {
    setRecorded(prev => {
      if (prev?.url) URL.revokeObjectURL(prev.url)
      return item
    })
  }

  const discardRecording = () => {
    setRecorded(prev => {
      if (prev?.url) URL.revokeObjectURL(prev.url)
      return null
    })
  }

  const modeBtn = (active) =>
    `flex items-center justify-center gap-1.5 flex-1 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 ${
      active
        ? 'bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 shadow-sm'
        : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
    }`

  return (
    <div
      className="flex-1 flex flex-col h-full min-h-0 p-4 gap-3 bg-slate-100 dark:bg-slate-950 transition-colors duration-300"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        className="hidden"
      />

      {/* Chọn nguồn clip: tải lên hoặc tự quay */}
      <div className="shrink-0 flex p-1 rounded-xl bg-slate-200/60 dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800">
        <button onClick={() => setMode('upload')} className={modeBtn(mode === 'upload')}>
          <UploadCloud size={13} />
          <span>Tải video lên</span>
        </button>
        <button onClick={() => setMode('record')} className={modeBtn(mode === 'record')}>
          <Video size={13} />
          <span>Tự quay video</span>
        </button>
      </div>

      {mode === 'upload' ? (
        video ? (
          <>
            {/* Thanh tệp gọn — nhường tối đa không gian cho khung hình */}
            <div className="shrink-0 flex items-center justify-between gap-3 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-900 bg-white/50 dark:bg-slate-950/30">
              <div className="flex items-center gap-2 min-w-0">
                <FileVideo size={15} className="text-indigo-500 shrink-0" />
                <span className="text-xs font-semibold text-slate-700 dark:text-slate-200 truncate" title={video.name}>
                  {video.name}
                </span>
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:text-indigo-500 hover:border-indigo-500/30 text-xs font-semibold transition-all shrink-0"
              >
                <RefreshCw size={12} />
                <span>Tải video khác</span>
              </button>
            </div>

            <UploadPlayer key={video.id} item={video} source="upload" />
          </>
        ) : (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 min-h-0 flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed border-slate-300 dark:border-slate-800 hover:border-indigo-500/50 bg-white/30 dark:bg-slate-950/20 cursor-pointer hover:bg-white/50 dark:hover:bg-slate-900/10 transition-all duration-300 text-center px-8"
          >
            <div className="p-5 rounded-full bg-indigo-500/5 text-indigo-500 dark:text-indigo-400 border border-indigo-500/10">
              <UploadCloud size={40} className="animate-bounce" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                Kéo thả video vào đây hoặc <span className="text-indigo-500 hover:underline">chọn tệp</span>
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
                Hỗ trợ MP4, WEBM, MOV. Video sẽ được phát và dịch ngay trong lúc chạy.
              </p>
            </div>
          </div>
        )
      ) : recorded ? (
        <>
          <div className="shrink-0 flex items-center justify-between gap-3 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-900 bg-white/50 dark:bg-slate-950/30">
            <div className="flex items-center gap-2 min-w-0">
              <Clapperboard size={15} className="text-indigo-500 shrink-0" />
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-200 truncate">
                Clip vừa quay · {recorded.duration.toFixed(1)}s
              </span>
            </div>
            <button
              onClick={discardRecording}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:text-indigo-500 hover:border-indigo-500/30 text-xs font-semibold transition-all shrink-0"
            >
              <RefreshCw size={12} />
              <span>Quay lại clip khác</span>
            </button>
          </div>

          <UploadPlayer key={recorded.id} item={recorded} source="record" autoStart />
        </>
      ) : (
        <VideoRecorder onRecorded={acceptRecording} />
      )}
    </div>
  )
}
