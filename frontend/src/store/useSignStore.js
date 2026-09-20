import { create } from 'zustand'
import { revokeAudioUrl } from '../utils/audioHelper'

// 3 nguồn nhận dạng ĐỘC LẬP: realtime / upload / record.
// Kết quả (glosses/câu/audio/lịch sử) của mỗi nguồn tách riêng, không đè lên nhau —
// người dùng có thể ký realtime rồi bấm sang upload/record để so sánh mà không mất KQ.
const emptyResult = () => ({
  glosses: [],       // [{ id, text, timestamp }]
  sentence: '',
  history: [],       // câu đã dịch (trong phiên)
  audioUrl: null,    // blob URL (WAV)
  isProcessing: false,
  noSign: false,     // phiên vừa rồi không nhận được ký hiệu nào đủ tin cậy
  finishedAt: 0,     // mốc backend chốt xong phiên (đổi giá trị = tín hiệu "xong")
})

const useSignStore = create((set, get) => ({
  // -- WebSocket (dùng chung) --
  wsStatus: 'disconnected', // 'disconnected' | 'connecting' | 'connected' | 'error'
  setWsStatus: (status) => set({ wsStatus: status }),

  // -- Nguồn đang hiển thị ở ResultPanel --
  activeSource: 'realtime',
  setActiveSource: (source) => set({ activeSource: source }),

  // Nguồn nhận kết quả từ WebSocket: 'realtime' khi ký trước camera, 'upload' khi đang phát
  // video. Sai nguồn thì kết quả rơi nhầm bucket và panel tương ứng trống trơn.
  wsSink: 'realtime',
  setWsSink: (source) => set({ wsSink: source }),

  // -- Kết quả theo từng nguồn --
  sources: {
    realtime: emptyResult(),
    upload: emptyResult(),
    record: emptyResult(),
  },

  addGloss: (source, text) => set(s => {
    const cur = s.sources[source]
    return {
      sources: {
        ...s.sources,
        [source]: {
          ...cur,
          glosses: [...cur.glosses, { id: Date.now(), text, timestamp: Date.now() }],
          noSign: false, // có từ mới -> cảnh báo "không nhận được ký hiệu" hết hiệu lực
        },
      },
    }
  }),

  setSentence: (source, sentence) => set(s => {
    const cur = s.sources[source]
    let history = cur.history
    if (sentence && sentence.trim() !== '') {
      const last = history[history.length - 1]
      if (last !== sentence) history = [...history, sentence]
    }
    return { sources: { ...s.sources, [source]: { ...cur, sentence, history } } }
  }),

  setProcessing: (source, v) => set(s => ({
    sources: { ...s.sources, [source]: { ...s.sources[source], isProcessing: v } },
  })),

  setAudioUrl: (source, url) => set(s => {
    const prev = s.sources[source].audioUrl
    if (prev) revokeAudioUrl(prev) // tránh rò rỉ bộ nhớ, chỉ thu hồi audio CỦA nguồn này
    return { sources: { ...s.sources, [source]: { ...s.sources[source], audioUrl: url } } }
  }),

  // Backend đã chốt xong phiên (có câu + giọng nói, hoặc không nhận được ký hiệu nào).
  // Luôn được gọi -> chỗ nào đang chờ kết quả cũng thoát được, không treo tới hết timeout.
  markFinished: (source, { noSign = false } = {}) => set(s => ({
    sources: {
      ...s.sources,
      [source]: { ...s.sources[source], noSign, isProcessing: false, finishedAt: Date.now() },
    },
  })),

  // Lượt dịch mới: xoá ký hiệu/câu/giọng nói của lượt trước nhưng giữ lịch sử.
  startNewRun: (source) => set(s => {
    const cur = s.sources[source]
    if (cur.audioUrl) revokeAudioUrl(cur.audioUrl)
    return {
      sources: {
        ...s.sources,
        [source]: {
          ...cur,
          glosses: [],
          sentence: '',
          audioUrl: null,
          isProcessing: false,
          noSign: false,
          finishedAt: 0,
        },
      },
    }
  }),

  // Xóa kết quả của MỘT nguồn (không đụng các nguồn khác).
  clearSession: (source) => set(s => {
    const prev = s.sources[source].audioUrl
    if (prev) revokeAudioUrl(prev)
    return { sources: { ...s.sources, [source]: emptyResult() } }
  }),

  // -- Cài đặt & toggle (dùng chung, lưu localStorage) --
  theme: localStorage.getItem('sign_speak_theme') || 'dark', // 'dark' | 'light'
  voiceGender: localStorage.getItem('sign_speak_voice_gender') || 'woman', // 'man' | 'woman'
  voiceSpeed: parseFloat(localStorage.getItem('sign_speak_voice_speed')) || 1.0,

  toggleTheme: () => {
    const nextTheme = get().theme === 'dark' ? 'light' : 'dark'
    localStorage.setItem('sign_speak_theme', nextTheme)
    set({ theme: nextTheme })
  },

  toggleVoiceGender: () => {
    const nextGender = get().voiceGender === 'woman' ? 'man' : 'woman'
    localStorage.setItem('sign_speak_voice_gender', nextGender)
    set({ voiceGender: nextGender })
  },

  cycleVoiceSpeed: () => {
    const speeds = [0.8, 1.0, 1.2, 1.5, 2.0]
    const currentSpeed = get().voiceSpeed
    const nextIndex = (speeds.indexOf(currentSpeed) + 1) % speeds.length
    const nextSpeed = speeds[nextIndex]
    localStorage.setItem('sign_speak_voice_speed', nextSpeed.toString())
    set({ voiceSpeed: nextSpeed })
  },

  // -- Camera Power --
  isCameraOn: true,
  setCameraOn: (on) => set({ isCameraOn: on }),

  // -- Debug Skeleton --
  showSkeleton: false,
  toggleSkeleton: () => set(s => ({ showSkeleton: !s.showSkeleton })),
}))

export default useSignStore
