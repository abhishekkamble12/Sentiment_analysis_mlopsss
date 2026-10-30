import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from 'recharts'
import type { Sentiment, TrendPoint } from '../types'
import { SENTIMENT_META } from './ui'

interface DonutDatum {
  name: Sentiment
  value: number
}

export function SentimentDonut({ data }: { data: Record<Sentiment, number> }) {
  const rows: DonutDatum[] = (['positive', 'neutral', 'negative'] as Sentiment[])
    .map((s) => ({ name: s, value: data[s] ?? 0 }))
  const total = rows.reduce((sum, r) => sum + r.value, 0)

  if (total === 0) {
    return <div className="h-[240px] flex items-center justify-center text-sm text-slate-500">No predictions yet</div>
  }

  return (
    <div className="h-[240px]">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={rows}
            dataKey="value"
            nameKey="name"
            innerRadius={62}
            outerRadius={92}
            paddingAngle={2}
            stroke="none"
          >
            {rows.map((r) => (
              <Cell key={r.name} fill={SENTIMENT_META[r.name].hex} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 12, color: '#e2e8f0' }}
            formatter={(value: number, _name: string, entry) => {
              const pct = total ? ((value / total) * 100).toFixed(1) : '0'
              const label = SENTIMENT_META[(entry?.payload as DonutDatum)?.name]?.label ?? ''
              return [`${value} (${pct}%)`, label]
            }}
          />
          <Legend
            formatter={(value) => <span className="text-slate-300 text-xs capitalize">{value}</span>}
            iconType="circle"
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

export function TrendAreaChart({ trends }: {
  trends: { categories: string[]; positive: number[]; neutral: number[]; negative: number[] }
}) {
  if (!trends.categories.length) {
    return <div className="h-[260px] flex items-center justify-center text-sm text-slate-500">No trend data yet</div>
  }
  const data = trends.categories.map((cat, i) => ({
    bucket: cat,
    Positive: trends.positive[i] ?? 0,
    Neutral: trends.neutral[i] ?? 0,
    Negative: trends.negative[i] ?? 0,
  }))

  const series: { key: 'Positive' | 'Neutral' | 'Negative'; color: string }[] = [
    { key: 'Positive', color: SENTIMENT_META.positive.hex },
    { key: 'Neutral', color: SENTIMENT_META.neutral.hex },
    { key: 'Negative', color: SENTIMENT_META.negative.hex },
  ]

  return (
    <div className="h-[260px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <defs>
            {series.map((s) => (
              <linearGradient key={s.key} id={`grad-${s.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={s.color} stopOpacity={0.35} />
                <stop offset="100%" stopColor={s.color} stopOpacity={0.02} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis dataKey="bucket" stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 12, color: '#e2e8f0' }}
          />
          <Legend formatter={(v) => <span className="text-slate-300 text-xs">{v}</span>} iconType="circle" />
          {series.map((s) => (
            <Area
              key={s.key}
              type="monotone"
              dataKey={s.key}
              stackId="1"
              stroke={s.color}
              strokeWidth={2}
              fill={`url(#grad-${s.key})`}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Simple horizontal bar comparison of macro-F1 across models. */
export function F1BarChart({ data }: { data: { model: string; macro_f1: number }[] }) {
  const max = Math.max(...data.map((d) => d.macro_f1), 0.001)
  return (
    <div className="space-y-2">
      {data.map((d) => {
        const pct = (d.macro_f1 / max) * 100
        return (
          <div key={d.model} className="flex items-center gap-3">
            <span className="text-xs text-slate-400 w-40 truncate text-right">{d.model}</span>
            <div className="flex-1 h-5 bg-slate-800/70 rounded-md overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-md transition-all duration-700"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="text-xs font-mono text-slate-200 w-12">{d.macro_f1.toFixed(3)}</span>
          </div>
        )
      })}
    </div>
  )
}

const TOPIC_SERIES: { key: 'positive' | 'neutral' | 'negative' | 'uncertain'; label: string; color: string }[] = [
  { key: 'positive', label: 'Positive', color: SENTIMENT_META.positive.hex },
  { key: 'neutral', label: 'Neutral', color: SENTIMENT_META.neutral.hex },
  { key: 'negative', label: 'Negative', color: SENTIMENT_META.negative.hex },
  { key: 'uncertain', label: 'Uncertain', color: '#fbbf24' },
]

/** Posts per day for a topic, stacked by the model's label. */
export function TopicTrendChart({ points }: { points: TrendPoint[] }) {
  if (!points.length) {
    return <div className="h-[240px] flex items-center justify-center text-sm text-slate-500">No posts in this period</div>
  }
  const data = points.map((p) => ({ bucket: p.bucket.slice(5), ...p.counts }))
  return (
    <div className="h-[240px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis dataKey="bucket" stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 12, color: '#e2e8f0' }}
          />
          <Legend formatter={(v) => <span className="text-slate-300 text-xs">{v}</span>} iconType="circle" />
          {TOPIC_SERIES.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} stackId="posts" fill={s.color} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
