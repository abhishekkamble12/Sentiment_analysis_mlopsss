import { useEffect, useState } from 'react'
import {
  LayoutDashboard,
  MessageSquareText,
  FileSpreadsheet,
  FlaskConical,
  History as HistoryIcon,
  Activity,
  Bot,
  Github,
  KeyRound,
  Radar,
  Tags,
} from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import CsvAnalysis from './pages/CsvAnalysis'
import ModelLab from './pages/ModelLab'
import History from './pages/History'
import Topics from './pages/Topics'
import Agent from './pages/Agent'
import Annotate from './pages/Annotate'
import { api, keyStore } from './api/client'

type PageId = 'dashboard' | 'predict' | 'csv' | 'modellab' | 'history' | 'topics' | 'agent' | 'annotate'

const NAV: { id: PageId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'predict', label: 'Text Prediction', icon: MessageSquareText },
  { id: 'csv', label: 'Batch CSV Analysis', icon: FileSpreadsheet },
  { id: 'modellab', label: 'Model Lab', icon: FlaskConical },
  { id: 'history', label: 'Prediction History', icon: HistoryIcon },
  { id: 'topics', label: 'Topic Monitor', icon: Radar },
  { id: 'agent', label: 'Ask the Agent', icon: Bot },
  { id: 'annotate', label: 'Annotation Queue', icon: Tags },
]

export default function App() {
  const [page, setPage] = useState<PageId>('dashboard')
  const [health, setHealth] = useState<'ok' | 'down' | 'loading'>('loading')

  const checkHealth = () => {
    api
      .health()
      .then(() => setHealth('ok'))
      .catch(() => setHealth('down'))
  }
  useEffect(checkHealth, [])

  const activePage = NAV.find((n) => n.id === page)!
  const PageIcon = activePage.icon

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-slate-950">
      {/* Sidebar — vertical on desktop, horizontal scroll on mobile */}
      <aside className="md:w-64 md:min-h-screen md:border-r border-slate-800 bg-slate-900/50 backdrop-blur-sm flex flex-col">
        <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-indigo-700 flex items-center justify-center flex-shrink-0">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div className="min-w-0">
            <h1 className="font-bold text-white text-sm leading-tight truncate">Sentiment Analysis</h1>
            <p className="text-xs text-slate-400 truncate">Tweet Classifier</p>
          </div>
        </div>

        <nav className="flex md:flex-col gap-1 px-3 py-3 overflow-x-auto md:overflow-visible flex-1">
          {NAV.map((item) => {
            const Icon = item.icon
            const active = item.id === page
            return (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all whitespace-nowrap ${active
                    ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                  }`}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                <span>{item.label}</span>
              </button>
            )
          })}
        </nav>

        <ApiKeyField />

        <div className="hidden md:flex items-center gap-2 px-5 py-4 border-t border-slate-800 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${health === 'ok' ? 'bg-emerald-400' : health === 'down' ? 'bg-rose-400' : 'bg-amber-400 animate-pulse'
              }`}
          />
          <span className="text-slate-400">
            {health === 'ok' ? 'Backend online' : health === 'down' ? 'Backend unreachable' : 'Checking…'}
          </span>
          <a
            href="https://github.com/abhishekkamble12/Sentiment_analysis_mlopsss"
            target="_blank"
            rel="noreferrer"
            className="ml-auto text-slate-500 hover:text-slate-300"
            aria-label="Repository"
          >
            <Github className="w-4 h-4" />
          </a>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 min-w-0">
        <div className="px-5 md:px-8 py-5 md:py-7">
          <div className="flex items-center gap-3 mb-6">
            <PageIcon className="w-6 h-6 text-indigo-400" />
            <div>
              <h2 className="text-xl font-bold text-white">{activePage.label}</h2>
            </div>
          </div>
          {page === 'dashboard' && <Dashboard />}
          {page === 'predict' && <Predict />}
          {page === 'csv' && <CsvAnalysis />}
          {page === 'modellab' && <ModelLab />}
          {page === 'history' && <History />}
          {page === 'topics' && <Topics />}
          {page === 'agent' && <Agent />}
          {page === 'annotate' && <Annotate />}
        </div>
      </main>
    </div>
  )
}

/** The key for the protected actions (collect, rebuild, agent, topic writes). Kept in sessionStorage only. */
function ApiKeyField() {
  const [value, setValue] = useState(keyStore.get())
  return (
    <div className="hidden md:block px-4 pb-3">
      <label className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
        <KeyRound className="w-3.5 h-3.5" /> API key
      </label>
      <input
        type="password"
        autoComplete="off"
        value={value}
        onChange={(e) => { setValue(e.target.value); keyStore.set(e.target.value.trim()) }}
        placeholder="needed for protected actions"
        className="input-base !py-2 !text-xs"
      />
    </div>
  )
}
