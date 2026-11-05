import { useEffect, useState } from 'react'
import CameraPanel from './components/CameraPanel/CameraPanel'
import ResultPanel from './components/ResultPanel/ResultPanel'
import LandingPage from './components/LandingPage/LandingPage'
import StatusBar from './components/CameraPanel/StatusBar'
import useSignStore from './store/useSignStore'
import wsClient from './services/ws'
import logoDark from './assets/logo_dark.png'
import logoLight from './assets/logo_light.png'


export default function App() {
  const wsStatus = useSignStore(s => s.wsStatus)
  const setWsStatus = useSignStore(s => s.setWsStatus)
  const addGloss = useSignStore(s => s.addGloss)
  const setSentence = useSignStore(s => s.setSentence)
  const setAudioUrl = useSignStore(s => s.setAudioUrl)
  const setProcessing = useSignStore(s => s.setProcessing)
  const markFinished = useSignStore(s => s.markFinished)
  const setActiveSource = useSignStore(s => s.setActiveSource)
  const theme = useSignStore(s => s.theme)

  const [activeTab, setActiveTab] = useState('realtime')
  const [view, setView] = useState(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      return params.get('view') === 'demo' ? 'demo' : 'landing'
    }
    return 'landing'
  })

  // Listen to browser forward/back button clicks
  useEffect(() => {
    const handlePopState = () => {
      if (typeof window !== 'undefined') {
        const params = new URLSearchParams(window.location.search)
        setView(params.get('view') === 'demo' ? 'demo' : 'landing')
      }
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  // Sync theme selection to documentElement root class and favicon
  useEffect(() => {
    const root = document.documentElement
    
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }

    // Dynamic browser tab favicon change based on active theme (recreate node to bypass browser caching)
    const existingFavicon = document.querySelector("link[rel*='icon']")
    if (existingFavicon) {
      existingFavicon.remove()
    }
    const newFavicon = document.createElement('link')
    newFavicon.rel = 'icon'
    newFavicon.type = 'image/png'
    newFavicon.href = theme === 'dark' ? logoLight : logoDark
    document.head.appendChild(newFavicon)
  }, [theme])

  // realtime tab open -> ResultPanel shows 'realtime' source results; manual tabs set
  // activeSource to 'upload'/'record' themselves.
  useEffect(() => {
    if (activeTab === 'realtime') setActiveSource('realtime')
  }, [activeTab, setActiveSource])

  useEffect(() => {
    if (view !== 'demo') return

    // Write to the currently streaming source; read getState() per message (not a stale
    // closure) so the source is always current.
    const sink = () => useSignStore.getState().wsSink

    wsClient.connect({
      onStatus: setWsStatus,
      onGloss: (g) => addGloss(sink(), g),
      onSentence: (s) => setSentence(sink(), s),
      onAudio: (url) => setAudioUrl(sink(), url),
      onProcessing: (v) => setProcessing(sink(), v),
      onFinished: (info) => markFinished(sink(), info),
    })

    return () => {
      wsClient.disconnect()
    }
  }, [view, setWsStatus, addGloss, setSentence, setAudioUrl, setProcessing, markFinished])

  const startDemo = () => {
    window.history.pushState({}, '', '?view=demo')
    setView('demo')
  }

  const goBackToLanding = () => {
    window.history.pushState({}, '', window.location.pathname)
    setView('landing')
  }

  if (view === 'landing') {
    return <LandingPage onStart={startDemo} />
  }

  return (
    <div className="relative flex flex-col h-screen w-screen overflow-hidden bg-slate-50 text-slate-800 dark:bg-slate-950 dark:text-slate-100 font-sans transition-colors duration-300">
      
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a04_1px,transparent_1px),linear-gradient(to_bottom,#0f172a04_1px,transparent_1px)] dark:bg-[linear-gradient(to_right,#ffffff02_1px,transparent_1px),linear-gradient(to_bottom,#ffffff02_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none" />

      <div className="absolute top-[-100px] left-[-100px] ambient-glow pointer-events-none" />
      <div className="absolute bottom-[-150px] right-[-150px] ambient-glow pointer-events-none" />

      <StatusBar status={wsStatus} activeTab={activeTab} setActiveTab={setActiveTab} onBack={goBackToLanding} />

      <div className="relative z-10 flex flex-col md:flex-row flex-1 w-full min-h-0 overflow-hidden">
        <main className="w-full md:w-2/3 h-1/2 md:h-full flex flex-col min-h-0">
          <CameraPanel activeTab={activeTab} />
        </main>

        <aside className="w-full md:w-1/3 h-1/2 md:h-full flex flex-col min-h-0">
          <ResultPanel />
        </aside>
      </div>
    </div>
  )
}



