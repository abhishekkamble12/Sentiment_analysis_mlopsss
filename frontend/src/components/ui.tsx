import type { ReactNode } from 'react'
import { AlertTriangle, Inbox } from 'lucide-react'
import type { Sentiment } from '../types'

export const SENTIMENT_META: Record<Sentiment, {
  label: string
  text: string
  badge: string
  dot: string
  hex: string
}> = {
  positive: { label: 'Positive', text: 'text-emerald-400', badge: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30', dot: 'bg-emerald-400', hex: '#10b981' },
  neutral: { label: 'Neutral', text: 'text-slate-300', badge: 'bg-slate-500/10 text-slate-300 border-slate-500/30', dot: 'bg-slate-400', hex: '#94a3b8' },
  negative: { label: 'Negative', text: 'text-rose-400', badge: 'bg-rose-500/10 text-rose-400 border-rose-500/30', dot: 'bg-rose-400', hex: '#f43f5e' },
}

const SENTIMENT_ORDER: Sentiment[] = ['positive', 'neutral', 'negative']

export function normaliseSentiment(value: string | null | undefined): Sentiment {
  const v = (value || '').toLowerCase()
  return (SENTIMENT_ORDER as string[]).includes(v) ? (v as Sentiment) : 'neutral'
}

export function SentimentBadge({ sentiment, className = '' }: { sentiment: Sentiment; className?: string }) {
  const m = SENTIMENT_META[sentiment]
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-md border ${m.badge} ${className}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${m.dot}`} />
      {m.label}
    </span>
  )
}

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`glass-card p-5 ${className}`}>{children}</div>
}

export function SectionTitle({ icon, title, subtitle }: { icon?: ReactNode; title: string; subtitle?: string }) {
  return (
    <div className="flex items-start gap-3 mb-4">
      {icon && <div className="text-indigo-400 mt-0.5">{icon}</div>}
      <div>
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        {subtitle && <p className="text-sm text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
    </div>
  )
}

export function StatCard({
  label, value, sub, accent = 'indigo', icon,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  accent?: 'indigo' | 'emerald' | 'rose' | 'slate' | 'amber'
  icon?: ReactNode
}) {
  const ring: Record<string, string> = {
    indigo: 'text-indigo-400',
    emerald: 'text-emerald-400',
    rose: 'text-rose-400',
    slate: 'text-slate-300',
    amber: 'text-amber-400',
  }
  return (
    <Card className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-slate-400">{label}</span>
        {icon && <span className={ring[accent]}>{icon}</span>}
      </div>
      <span className="text-2xl font-bold text-white">{value}</span>
      {sub && <span className="text-xs text-slate-400">{sub}</span>}
    </Card>
  )
}

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-slate-800/70 ${className}`} />
}

export function Spinner({ className = 'w-5 h-5' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8V0C8.477 0 4 4.477 4 10z" />
    </svg>
  )
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-12 px-4">
      <Inbox className="w-10 h-10 text-slate-600 mb-3" />
      <p className="text-slate-300 font-medium">{title}</p>
      {hint && <p className="text-sm text-slate-500 mt-1 max-w-sm">{hint}</p>}
    </div>
  )
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-start gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm">
      <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <p>{message}</p>
        {onRetry && (
          <button onClick={onRetry} className="mt-2 text-rose-200 underline hover:text-white">
            Try again
          </button>
        )}
      </div>
    </div>
  )
}

export function ConfidenceBar({ confidence }: { confidence: number | null }) {
  if (confidence === null) {
    return <span className="text-xs text-slate-500">not reported</span>
  }
  const pct = Math.round(confidence * 100)
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-slate-400">Confidence</span>
        <span className="font-mono text-cyan-400">{pct}%</span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
        <div className="h-2.5 rounded-full bg-gradient-to-r from-cyan-500 to-indigo-500 transition-all duration-700" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return Number(v).toFixed(digits)
}

export function fmtTime(ts: string | null): string {
  if (!ts) return '—'
  try {
    const d = new Date(ts.endsWith('Z') ? ts : ts + 'Z')
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch {
    return ts
  }
}
