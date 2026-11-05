import { create } from 'zustand'
import { revokeAudioUrl } from '../utils/audioHelper'

// 3 independent recognition sources: realtime / upload / record. Each keeps its own
// glosses/sentence/audio/history so switching tabs doesn't overwrite or lose results.
const emptyResult = () => ({
  glosses: [],       // [{ id, text, timestamp }]
  sentence: '',
  history: [],       // translated sentences (this session)
  audioUrl: null,    // blob URL (WAV)
  isProcessing: false,
  noSign: false,     // true when the last session got no confident sign
  finishedAt: 0,     // timestamp the backend finished the session (value change = "done" signal)
})

const useSignStore = create((set, get) => ({
  wsStatus: 'disconnected', // 'disconnected' | 'connecting' | 'connected' | 'error'
  setWsStatus: (status) => set({ wsStatus: status }),

  activeSource: 'realtime',
  setActiveSource: (source) => set({ activeSource: source }),

  // Source receiving WS results: 'realtime' while signing to camera, 'upload' while playing a
  // video. Wrong value routes results to the wrong bucket, leaving that panel empty.
  wsSink: 'realtime',
  setWsSink: (source) => set({ wsSink: source }),

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
          noSign: false, // new gloss arrived -> clear the "no sign recognized" warning
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
    if (prev) revokeAudioUrl(prev) // avoid leaking memory; only revoke this source's audio
    return { sources: { ...s.sources, [source]: { ...s.sources[source], audioUrl: url } } }
  }),

  // Backend finished the session (sentence + audio, or no confident sign). Always called so
  // anything awaiting a result can exit instead of hanging until timeout.
  markFinished: (source, { noSign = false } = {}) => set(s => ({
    sources: {
      ...s.sources,
      [source]: { ...s.sources[source], noSign, isProcessing: false, finishedAt: Date.now() },
    },
  })),

  // New run: clears the previous glosses/sentence/audio but keeps history.
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

  // Clears one source's results without touching the others.
  clearSession: (source) => set(s => {
    const prev = s.sources[source].audioUrl
    if (prev) revokeAudioUrl(prev)
    return { sources: { ...s.sources, [source]: emptyResult() } }
  }),

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

  isCameraOn: true,
  setCameraOn: (on) => set({ isCameraOn: on }),

  showSkeleton: false,
  toggleSkeleton: () => set(s => ({ showSkeleton: !s.showSkeleton })),
}))

export default useSignStore
