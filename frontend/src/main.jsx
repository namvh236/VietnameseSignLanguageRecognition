import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

const requiredEnvVars = ['VITE_WS_URL', 'VITE_API_URL']
requiredEnvVars.forEach(key => {
  if (!import.meta.env[key]) {
    console.error(`[Configuration Error] Missing required env var: ${key}`)
    throw new Error(`Missing required environment variable: ${key}`)
  }
})


createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
