import useSignStore from '../../store/useSignStore'
import GlossBadgeList from './GlossBadgeList'
import SentenceDisplay from './SentenceDisplay'
import AudioPlayer from './AudioPlayer'
import { Trash2, MessageSquare, Venus, Mars, AlertTriangle } from 'lucide-react'

const SOURCE_LABELS = {
  realtime: 'Realtime',
  upload: 'Tải lên',
  record: 'Tự ghi hình',
}

export default function ResultPanel() {
  // Hiển thị kết quả của NGUỒN đang mở (realtime/upload/record) — mỗi nguồn giữ KQ riêng.
  const activeSource = useSignStore(s => s.activeSource)
  const result = useSignStore(s => s.sources[s.activeSource])
  const clearSession = useSignStore(s => s.clearSession)

  const { glosses, sentence, audioUrl, isProcessing, history, noSign } = result

  const voiceGender = useSignStore(s => s.voiceGender)
  const toggleVoiceGender = useSignStore(s => s.toggleVoiceGender)
  const voiceSpeed = useSignStore(s => s.voiceSpeed)
  const cycleVoiceSpeed = useSignStore(s => s.cycleVoiceSpeed)

  const hasAnyData = glosses.length > 0 || sentence || audioUrl || isProcessing || noSign

  return (
    <div className="flex flex-col h-full p-6 bg-white/70 dark:bg-slate-950/40 backdrop-blur-md border-t md:border-t-0 md:border-l border-slate-200 dark:border-slate-900 transition-colors duration-300">
      
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-6 border-b border-slate-200 dark:border-slate-900">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-500 dark:text-indigo-400">
            <MessageSquare size={18} />
          </div>
          <div className="flex flex-col">
            <h2 className="text-base font-bold text-slate-700 dark:text-slate-200 uppercase tracking-wider">
              Kết quả nhận dạng
            </h2>
            <span className="text-[10px] font-semibold text-indigo-500 dark:text-indigo-400 uppercase tracking-wide">
              {SOURCE_LABELS[activeSource] || activeSource}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {hasAnyData && (
            <button
              onClick={() => clearSession(activeSource)}
              className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl text-sm font-bold text-slate-500 hover:text-rose-600 dark:text-slate-400 dark:hover:text-rose-400 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-rose-900/30 hover:bg-rose-500/10 dark:hover:bg-rose-950/10 transition-all duration-200"
            >
              <Trash2 size={14} />
              <span>Xóa</span>
            </button>
          )}



          {/* Voice Gender Toggle Button */}
          <button
            onClick={toggleVoiceGender}
            className="p-2.5 rounded-xl text-slate-500 hover:text-indigo-500 dark:text-slate-400 dark:hover:text-indigo-400 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-indigo-500/30 transition-all duration-200 shadow-sm"
            title={voiceGender === 'woman' ? 'Giọng đọc: Nữ (Click để đổi sang Nam)' : 'Giọng đọc: Nam (Click để đổi sang Nữ)'}
          >
            {voiceGender === 'woman' ? <Venus size={20} className="text-pink-500 dark:text-pink-400" /> : <Mars size={20} className="text-sky-500 dark:text-sky-400" />}
          </button>

          {/* Voice Speed Cycle Button */}
          <button
            onClick={cycleVoiceSpeed}
            className="px-3 py-2.5 rounded-xl text-slate-500 hover:text-indigo-500 dark:text-slate-400 dark:hover:text-indigo-400 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-indigo-500/30 transition-all duration-200 flex items-center justify-center min-w-[54px] shadow-sm"
            title={`Tốc độ đọc: ${voiceSpeed.toFixed(1)}x (Click để thay đổi)`}
          >
            <span className="text-sm font-bold tracking-tight">{voiceSpeed.toFixed(1)}x</span>
          </button>
        </div>
      </div>

      {/* Panel Body Content */}
      <div className="flex-1 flex flex-col justify-start py-6 gap-6 overflow-y-auto">
        
        {/* Sign Language Glosses */}
        <section className="flex flex-col gap-2">
          <p className="text-sm font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Ký hiệu</p>
          <div className="p-3.5 rounded-2xl bg-slate-100/50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-900">
            <GlossBadgeList glosses={glosses} />
          </div>
        </section>

        {/* Hết phiên mà không ký hiệu nào đủ tin cậy: nói rõ ra, đừng để panel trống trơn */}
        {noSign && glosses.length === 0 && (
          <div className="flex items-start gap-2.5 px-4 py-3 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-400">
            <AlertTriangle size={16} className="shrink-0 mt-0.5" />
            <p className="text-xs font-semibold leading-relaxed">
              Không nhận được ký hiệu nào đủ tin cậy nên chưa ghép được câu. Hãy ký chậm và rõ
              hơn, giữ hai tay trong khung hình và hạ tay giữa các từ.
            </p>
          </div>
        )}

        {/* Full Translated Sentence */}
        <section className="flex flex-col gap-2">
          <p className="text-sm font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Bản dịch</p>
          <SentenceDisplay sentence={sentence} isProcessing={isProcessing} />
        </section>

        {/* Translation History */}
        {history.length > 0 && (
          <section className="flex flex-col gap-2 border-t border-slate-100 dark:border-slate-900 pt-6">
            <p className="text-sm font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Lịch sử dịch</p>
            <div className="flex flex-col gap-2 max-h-40 overflow-y-auto pr-1">
              {history.map((h, i) => (
                <div key={i} className="px-4 py-2.5 rounded-xl bg-slate-50/50 dark:bg-slate-900/25 border border-slate-200 dark:border-slate-800/40 text-sm text-slate-700 dark:text-slate-300 leading-normal">
                  {h}
                </div>
              ))}
            </div>
          </section>
        )}
      </div>

      {/* TTS Audio Player */}
      {audioUrl && (
        <div className="mt-auto pt-6 border-t border-slate-200 dark:border-slate-900 flex flex-col gap-4">
          <AudioPlayer audioUrl={audioUrl} />
        </div>
      )}
    </div>
  )
}
