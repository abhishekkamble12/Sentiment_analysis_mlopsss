import { useEffect, useState } from 'react'
import { Gauge, Timer, MessageCircle, TrendingUp, Cpu, AlertTriangle } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import type { Analytics, ModelInfo, RecentLog, Sentiment } from '../types'
import {
  Card, EmptyState, ErrorBanner, SentimentBadge, Skeleton, StatCard,
  fmtNum, fmtPct, fmtTime, normaliseSentiment,
} from '../components/ui'
import { SentimentDonut, TrendAreaChart } from '../components/charts'

export default function Dashboard() {
  const analytics = useAsync<Analytics>(() => api.analytics())
  const logs = useAsync<{ logs: RecentLog[] }>(() => api.recentLogs(8))
  const model = useAsync<ModelInfo>(() => api.modelInfo())

  const refreshAll = () => {
    analytics.refetch()
    logs.refetch()
    model.refetch()
  }

  return (
    <div className="space-y-6">
      {analytics.error && <ErrorBanner message={errorMessage(new Error(analytics.error))} onRetry={refreshAll} />}

      {/* Top stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {analytics.loading && !analytics.data ? (
          Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)
        ) : (
          <>
            <StatCard label="Total Analyzed" value={analytics.data?.total_insights ?? 0} icon={<MessageCircle className="w-5 h-5" />} accent="indigo" />
            <StatCard
              label="Avg Latency"
              value={`${fmtNum(analytics.data?.avg_latency_ms, 0)} ms`}
              icon={<Timer className="w-5 h-5" />}
              accent="emerald"
            />
            <StatCard
              label="Avg Confidence"
              value={fmtPct(analytics.data?.avg_confidence)}
              icon={<Gauge className="w-5 h-5" />}
              sub={analytics.data?.avg_confidence == null ? 'reported only by local models' : undefined}
              accent="amber"
            />
            <StatCard
              label="Sentiment Index"
              value={analytics.data?.sentiment_index == null ? '—' : Math.round(analytics.data.sentiment_index)}
              icon={<TrendingUp className="w-5 h-5" />}
              sub="0 (neg) → 100 (pos)"
              accent="indigo"
            />
          </>
        )}
      </div>

      {/* Distribution + trend */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="lg:col-span-1">
          <h3 className="text-sm font-semibold text-slate-300 mb-2">Sentiment Distribution</h3>
          {analytics.loading && !analytics.data ? (
            <Skeleton className="h-[240px]" />
          ) : (
            <SentimentDonut data={(analytics.data?.sentiment_distribution ?? { positive: 0, neutral: 0, negative: 0 }) as Record<Sentiment, number>} />
          )}
        </Card>

        <Card className="lg:col-span-2">
          <h3 className="text-sm font-semibold text-slate-300 mb-2">Sentiment Trends (hourly)</h3>
          {analytics.loading && !analytics.data ? (
            <Skeleton className="h-[260px]" />
          ) : (
            <TrendAreaChart trends={analytics.data?.trends ?? { categories: [], positive: [], neutral: [], negative: [] }} />
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Recent predictions */}
        <Card className="lg:col-span-2">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">Recent Predictions</h3>
          {logs.loading && !logs.data ? (
            <div className="space-y-2">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-12" />)}</div>
          ) : logs.data && logs.data.logs.length > 0 ? (
            <ul className="divide-y divide-slate-800">
              {logs.data.logs.map((log) => (
                <li key={log.id} className="flex items-center justify-between py-2.5">
                  <div className="min-w-0 flex items-center gap-3">
                    <SentimentBadge sentiment={normaliseSentiment(log.prediction)} />
                    <span className="text-xs text-slate-500 font-mono">{fmtTime(log.timestamp)}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <span className="text-slate-400">{log.model_used}</span>
                    <span className="font-mono text-slate-300">{log.latency_ms.toFixed(0)}ms</span>
                    <span className="font-mono text-slate-400 w-10 text-right">
                      {log.confidence == null ? '—' : fmtPct(log.confidence, 0)}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No predictions yet" hint="Submit text on the Text Prediction page to see it appear here." />
          )}
        </Card>

        {/* Model status */}
        <ModelStatus info={model.data} loading={model.loading} />
      </div>
    </div>
  )
}

function ModelStatus({ info, loading }: { info: ModelInfo | null; loading: boolean }) {
  const [tick, setTick] = useState(false)
  // animate the "ready" dot once
  useEffect(() => {
    if (info) setTick((t) => !t)
  }, [info])

  return (
    <Card>
      <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
        <Cpu className="w-4 h-4 text-indigo-400" /> Model & Processing Status
      </h3>
      {loading && !info ? (
        <Skeleton className="h-32" />
      ) : info ? (
        <dl className="space-y-3 text-sm">
          <Row label="Pipeline">
            <span className={`inline-flex items-center gap-1.5 ${info.pipeline_ready ? 'text-emerald-400' : 'text-rose-400'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${info.pipeline_ready ? 'bg-emerald-400' : 'bg-rose-400'}`} />
              {info.pipeline_ready ? 'Ready' : 'Not initialised'}
            </span>
          </Row>
          <Row label="Served model">
            <span className="font-mono text-slate-200">{info.best_model_name ?? '—'}</span>
          </Row>
          <Row label="Inference mode">
            <span className="font-mono text-indigo-300">{info.inference_mode ?? '—'}</span>
          </Row>
          <Row label="Transformer tier">
            <span className={info.transformer_enabled ? 'text-emerald-400' : 'text-slate-500'}>
              {info.transformer_enabled ? 'Enabled (3-class)' : 'Disabled (no checkpoint)'}
            </span>
          </Row>
          {info.best_metrics && (
            <Row label="Macro-F1 (silver)">
              <span className="font-mono text-slate-200">{info.best_metrics.macro_f1?.toFixed(4)}</span>
            </Row>
          )}
        </dl>
      ) : (
        <ErrorBanner message="Could not load model status" />
      )}
      <p className="mt-3 text-xs text-slate-500 flex items-start gap-1.5">
        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
        Metrics are silver (TextBlob-labelled), not human-labelled.
      </p>
    </Card>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-slate-400">{label}</dt>
      <dd className="text-right">{children}</dd>
    </div>
  )
}
