import { useEffect, useState } from 'react'
import { Search, ChevronLeft, ChevronRight } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import type { PredictionHistory, Sentiment } from '../types'
import { Card, EmptyState, ErrorBanner, SentimentBadge, Skeleton, fmtPct, fmtTime, normaliseSentiment } from '../components/ui'

const LIMIT = 15

export default function History() {
  const [q, setQ] = useState('')
  const [sentiment, setSentiment] = useState<string>('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [page, setPage] = useState(1)
  const [data, setData] = useState<PredictionHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Debounced search: refetch when filters/page change (q debounced via timeout)
  useEffect(() => {
    const t = setTimeout(() => {
      let cancelled = false
      setLoading(true)
      setError(null)
      api
        .predictionHistory({ q: q || undefined, sentiment: sentiment || undefined, start: start || undefined, end: end || undefined, page, limit: LIMIT })
        .then((res) => !cancelled && (setData(res), setLoading(false)))
        .catch((err) => !cancelled && (setError(errorMessage(err, 'Failed to load history')), setLoading(false)))
      return () => { cancelled = true }
    }, 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, sentiment, start, end, page])

  const resetPage = () => setPage(1)

  return (
    <div className="space-y-4">
      {/* Filters */}
      <Card>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              value={q}
              onChange={(e) => { setQ(e.target.value); resetPage() }}
              placeholder="Search text…"
              className="input-base pl-9"
            />
          </div>
          <select value={sentiment} onChange={(e) => { setSentiment(e.target.value); resetPage() }} className="input-base">
            <option value="">All sentiments</option>
            <option value="positive">Positive</option>
            <option value="neutral">Neutral</option>
            <option value="negative">Negative</option>
          </select>
          <input type="date" value={start} onChange={(e) => { setStart(e.target.value); resetPage() }} className="input-base" />
          <input type="date" value={end} onChange={(e) => { setEnd(e.target.value); resetPage() }} className="input-base" />
        </div>
      </Card>

      {error && <ErrorBanner message={error} />}

      {/* Table */}
      <Card>
        {loading ? (
          <div className="space-y-2">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-12" />)}</div>
        ) : data && data.logs.length > 0 ? (
          <>
            <div className="overflow-x-auto -mx-2 px-2">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-slate-500 border-b border-slate-800">
                    <th className="py-2 pr-4 font-medium">Text</th>
                    <th className="py-2 px-3 font-medium">Sentiment</th>
                    <th className="py-2 px-3 font-medium">Confidence</th>
                    <th className="py-2 px-3 font-medium">Model</th>
                    <th className="py-2 px-3 font-medium">Latency</th>
                    <th className="py-2 px-3 font-medium">Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {data.logs.map((log) => (
                    <tr key={log.id}>
                      <td className="py-2.5 pr-4 max-w-sm">
                        <p className="text-slate-200 truncate" title={log.text ?? ''}>
                          {log.text ?? <span className="text-slate-600 italic">text not stored</span>}
                        </p>
                      </td>
                      <td className="py-2.5 px-3"><SentimentBadge sentiment={normaliseSentiment(log.prediction) as Sentiment} /></td>
                      <td className="py-2.5 px-3 font-mono text-slate-300">{log.confidence == null ? '—' : fmtPct(log.confidence, 0)}</td>
                      <td className="py-2.5 px-3 font-mono text-xs text-slate-400">{log.model_used}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-300">{log.latency_ms.toFixed(0)}ms</td>
                      <td className="py-2.5 px-3 text-xs text-slate-400 whitespace-nowrap">{fmtTime(log.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-4 text-sm">
              <span className="text-slate-400">
                {data.total.toLocaleString()} result(s) · page {data.page} of {Math.max(data.total_pages, 1)}
              </span>
              <div className="flex gap-2">
                <button
                  className="btn-ghost disabled:opacity-40"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                >
                  <ChevronLeft className="w-4 h-4" /> Prev
                </button>
                <button
                  className="btn-ghost disabled:opacity-40"
                  onClick={() => setPage((p) => p + 1)}
                  disabled={page >= data.total_pages}
                >
                  Next <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </>
        ) : (
          <EmptyState title="No predictions match your filters" hint="Try clearing the search or date range, or predict some text first." />
        )}
      </Card>
    </div>
  )
}
