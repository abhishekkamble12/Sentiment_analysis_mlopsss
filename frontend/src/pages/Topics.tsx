import { useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import type { Alert, Drift, Insights, PostsResponse, Topic, TrendPoint } from '../types'
import { TopicTrendChart } from '../components/charts'
import {
  Card, EmptyState, ErrorBanner, PostBadge, Skeleton, Spinner, fmtPct, fmtTime,
} from '../components/ui'
import { useAsync } from '../hooks/useAsync'

type Tab = 'overview' | 'posts' | 'themes'

export default function Topics() {
  const topics = useAsync(() => api.topics(), [])
  const [selected, setSelected] = useState<number | null>(null)
  const [name, setName] = useState('')
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const create = async () => {
    setBusy(true)
    setError(null)
    try {
      const made = await api.createTopic({ name: name.trim(), query: query.trim() })
      setName('')
      setQuery('')
      topics.refetch()
      setSelected(made.id)
    } catch (err) {
      setError(errorMessage(err, 'Could not add the topic'))
    } finally {
      setBusy(false)
    }
  }

  const current = topics.data?.find((t) => t.id === selected) ?? null

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[20rem_1fr] gap-6">
      <div className="space-y-4">
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3">Track a topic</h3>
          <div className="space-y-2">
            <input className="input-base" placeholder="Name (e.g. Fuel prices)" maxLength={100} value={name}
              onChange={(e) => setName(e.target.value)} />
            <input className="input-base" placeholder="Search query" maxLength={200} value={query}
              onChange={(e) => setQuery(e.target.value)} />
            <button className="btn-primary w-full" disabled={busy || !name.trim() || !query.trim()} onClick={create}>
              {busy ? <Spinner /> : <Plus className="w-4 h-4" />} Add topic
            </button>
          </div>
          {error && <p className="mt-3 text-sm text-rose-400">{error}</p>}
          <p className="mt-3 text-xs text-slate-500">
            A background worker collects public news about each topic and classifies it. Labels come from a model
            trained on lexicon-labelled tweets, so treat them as rough.
          </p>
        </Card>

        <Card className="!p-3">
          {topics.loading && <Skeleton className="h-16" />}
          {topics.error && <ErrorBanner message={topics.error} onRetry={topics.refetch} />}
          {topics.data && topics.data.length === 0 && <EmptyState title="No topics yet" hint="Add one above." />}
          <div className="space-y-2">
            {topics.data?.map((t) => (
              <button key={t.id} onClick={() => setSelected(t.id)}
                className={`w-full text-left p-3 rounded-xl border transition-colors ${t.id === selected
                    ? 'border-indigo-500 bg-indigo-500/10' : 'border-slate-800 hover:border-slate-600'}`}>
                <p className="text-sm font-medium text-white">{t.name}</p>
                <p className="text-xs text-slate-400 truncate">{t.query}</p>
                <p className="text-xs text-slate-500 mt-1">
                  {t.posts} posts · {t.last_ingested ? `updated ${fmtTime(t.last_ingested)}` : 'not collected yet'}
                </p>
              </button>
            ))}
          </div>
        </Card>
      </div>

      <div className="min-w-0">
        {current ? (
          <TopicDetail key={current.id} topic={current}
            onDeleted={() => { setSelected(null); topics.refetch() }} onChanged={topics.refetch} />
        ) : (
          <Card><EmptyState title="Select a topic" hint="Its alerts, trends, posts and themes appear here." /></Card>
        )}
      </div>
    </div>
  )
}

function TopicDetail({ topic, onDeleted, onChanged }: { topic: Topic; onDeleted: () => void; onChanged: () => void }) {
  const [tab, setTab] = useState<Tab>('overview')
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null)
  const [working, setWorking] = useState<string | null>(null)

  const act = async (label: string, fn: () => Promise<unknown>, done: string) => {
    setWorking(label)
    setNotice(null)
    try {
      await fn()
      setNotice({ ok: true, text: done })
      onChanged()
    } catch (err) {
      setNotice({ ok: false, text: errorMessage(err, `${label} failed`) })
    } finally {
      setWorking(null)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-center gap-3">
          <div className="min-w-0 flex-1">
            <h3 className="text-lg font-semibold text-white">{topic.name}</h3>
            <p className="text-xs text-slate-400">query: {topic.query} · sources: {topic.sources.join(', ')}</p>
          </div>
          <button className="btn-ghost" disabled={!!working}
            onClick={() => act('Collect', () => api.ingestTopic(topic.id), 'Collection finished; refresh the tabs.')}>
            {working === 'Collect' ? <Spinner className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />} Collect now
          </button>
          <button className="btn-ghost !text-rose-300" disabled={!!working}
            onClick={() => window.confirm(`Delete "${topic.name}" and all its posts?`)
              && act('Delete', async () => { await api.deleteTopic(topic.id); onDeleted() }, 'Deleted.')}>
            <Trash2 className="w-4 h-4" /> Delete
          </button>
        </div>
        {notice && <p className={`mt-3 text-sm ${notice.ok ? 'text-emerald-400' : 'text-rose-400'}`}>{notice.text}</p>}
        <div className="mt-4 flex gap-2 border-b border-slate-800">
          {(['overview', 'posts', 'themes'] as Tab[]).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-3 py-2 text-sm capitalize -mb-px border-b-2 ${tab === t
                  ? 'border-indigo-500 text-white' : 'border-transparent text-slate-400 hover:text-slate-200'}`}>{t}</button>
          ))}
        </div>
      </Card>

      {tab === 'overview' && <Overview topicId={topic.id} />}
      {tab === 'posts' && <Posts topicId={topic.id} />}
      {tab === 'themes' && <Themes topicId={topic.id}
        onRebuild={() => act('Rebuild', () => api.runInsights(topic.id), 'Insights rebuilt.')} rebuilding={working === 'Rebuild'} />}
    </div>
  )
}

function Overview({ topicId }: { topicId: number }) {
  const alerts = useAsync<Alert[]>(() => api.alerts(topicId), [topicId])
  const drift = useAsync<Drift>(() => api.drift(topicId), [topicId])
  const trend = useAsync<TrendPoint[]>(() => api.trends(topicId), [topicId])

  return (
    <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
      <Card className="2xl:col-span-2">
        <h4 className="text-sm font-semibold text-white mb-2">Posts per day</h4>
        {trend.loading ? <Skeleton className="h-[240px]" /> : trend.error
          ? <ErrorBanner message={trend.error} onRetry={trend.refetch} /> : <TopicTrendChart points={trend.data ?? []} />}
      </Card>

      <Card>
        <h4 className="text-sm font-semibold text-white mb-2">Alerts</h4>
        {alerts.error && <ErrorBanner message={alerts.error} onRetry={alerts.refetch} />}
        {alerts.data?.length === 0 && (
          <p className="text-sm text-slate-500 italic">
            None. An alert needs at least 20 posts in the last day and in the week before.
          </p>
        )}
        <div className="space-y-2">
          {alerts.data?.map((a) => (
            <div key={a.id} className={`p-3 rounded-xl border text-sm ${a.acknowledged
                ? 'border-slate-800 text-slate-500' : 'border-amber-500/40 text-amber-200'}`}>
              <p>{a.message}</p>
              <div className="mt-1 flex items-center gap-3 text-xs text-slate-500">
                <span>{fmtTime(a.created_at)}</span>
                {!a.acknowledged && (
                  <button className="underline text-amber-300"
                    onClick={() => api.ackAlert(a.id).then(alerts.refetch)}>Acknowledge</button>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <h4 className="text-sm font-semibold text-white mb-2">
          Drift <span className="text-slate-500 font-normal">(last {drift.data?.window_hours ?? 24} h vs the week before)</span>
        </h4>
        {drift.loading && <Skeleton className="h-32" />}
        {drift.error && <ErrorBanner message={drift.error} onRetry={drift.refetch} />}
        {drift.data && (
          <dl className="text-sm text-slate-300 space-y-1.5">
            {[
              ['Posts', `${drift.data.window.n} vs ${drift.data.baseline.n}`],
              ['Unknown-word rate', `${fmtPct(drift.data.window.oov_rate)} vs ${fmtPct(drift.data.baseline.oov_rate)}`],
              ['Mean confidence', `${fmtPct(drift.data.window.mean_confidence)} vs ${fmtPct(drift.data.baseline.mean_confidence)}`],
              ['Uncertain posts', `${fmtPct(drift.data.window.uncertain_share)} vs ${fmtPct(drift.data.baseline.uncertain_share)}`],
              ['Class-mix shift (PSI)', drift.data.psi == null ? 'n/a' : drift.data.psi.toFixed(2)],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4"><dt>{k}</dt><dd className="font-mono text-slate-200">{v}</dd></div>
            ))}
            <p className="text-xs text-slate-500 pt-2">
              Unknown words are outside the model's vocabulary: a high value means this topic is far from the tweets the
              model was trained on. A PSI above 0.25 is a large shift in the positive/neutral/negative mix.
            </p>
          </dl>
        )}
      </Card>
    </div>
  )
}

function Posts({ topicId }: { topicId: number }) {
  const [sentiment, setSentiment] = useState('')
  const [page, setPage] = useState(1)
  const [data, setData] = useState<PostsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    api.posts(topicId, { sentiment: sentiment || undefined, page, limit: 15 })
      .then((res) => !cancelled && (setData(res), setLoading(false)))
      .catch((err) => !cancelled && (setError(errorMessage(err, 'Failed to load posts')), setLoading(false)))
    return () => { cancelled = true }
  }, [topicId, sentiment, page])

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-white">Collected posts {data && <span className="text-slate-500 font-normal">({data.total})</span>}</h4>
        <select className="input-base !w-auto !py-1.5 !text-sm" value={sentiment}
          onChange={(e) => { setSentiment(e.target.value); setPage(1) }}>
          <option value="">All labels</option>
          <option value="positive">Positive</option>
          <option value="neutral">Neutral</option>
          <option value="negative">Negative</option>
          <option value="uncertain">Uncertain</option>
        </select>
      </div>
      {error && <ErrorBanner message={error} />}
      {loading && !data && <Skeleton className="h-40" />}
      {data && data.posts.length === 0 && <EmptyState title="No posts" hint="The worker collects them on its next pass, or use Collect now." />}
      <div className="space-y-2">
        {data?.posts.map((p) => (
          <div key={p.id} className="p-3 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between gap-3 mb-1">
              <PostBadge label={p.prediction} />
              <span className="text-xs text-slate-500">{fmtTime(p.published_at)}</span>
            </div>
            {/* third-party text: rendered as text by React, never as HTML */}
            <p className="text-sm text-slate-300">{p.text}</p>
            <p className="text-xs text-slate-500 mt-1">
              {p.source}{p.confidence != null && ` · ${fmtPct(p.confidence, 0)} confident`}
              {p.url && <> · <a href={p.url} target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:underline">open</a></>}
            </p>
          </div>
        ))}
      </div>
      {data && data.total_pages > 1 && (
        <div className="flex items-center justify-between mt-4 text-sm text-slate-400">
          <button className="btn-ghost !py-1.5" disabled={page <= 1} onClick={() => setPage(page - 1)}><ChevronLeft className="w-4 h-4" /> Prev</button>
          <span>Page {data.page} of {data.total_pages}</span>
          <button className="btn-ghost !py-1.5" disabled={page >= data.total_pages} onClick={() => setPage(page + 1)}>Next <ChevronRight className="w-4 h-4" /></button>
        </div>
      )}
    </Card>
  )
}

function Themes({ topicId, onRebuild, rebuilding }: { topicId: number; onRebuild: () => void; rebuilding: boolean }) {
  const insights = useAsync<Insights>(() => api.insights(topicId), [topicId, rebuilding])

  return (
    <div className="space-y-4">
      {insights.loading && <Skeleton className="h-40" />}
      {insights.error && <ErrorBanner message={insights.error} onRetry={insights.refetch} />}
      {insights.data && (
        <>
          <Card>
            <div className="flex items-center justify-between gap-3">
              <h4 className="text-sm font-semibold text-white">Daily note</h4>
              <button className="btn-ghost !py-1.5" disabled={rebuilding} onClick={onRebuild}>
                {rebuilding ? <Spinner className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />} Rebuild
              </button>
            </div>
            <p className="text-xs text-slate-500 mb-2">
              {insights.data.llm.configured
                ? `Named by ${insights.data.llm.provider} / ${insights.data.llm.model}`
                : 'No LLM configured: themes are labelled by their top words and there is no daily note.'}
            </p>
            {insights.data.digest ? (
              <>
                <p className="text-sm text-slate-200">{insights.data.digest.summary}</p>
                {insights.data.digest.changes.length > 0 && (
                  <ul className="mt-2 list-disc list-inside text-sm text-slate-400">
                    {insights.data.digest.changes.map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                )}
              </>
            ) : <p className="text-sm text-slate-500 italic">No note yet.</p>}
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {insights.data.themes?.themes.map((t) => (
              <Card key={t.post_ids[0]}>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-semibold text-white">{t.label}</p>
                  <PostBadge label={t.sentiment} />
                </div>
                <p className="text-xs text-slate-500">{t.n} posts · {fmtPct(t.share, 0)} of the last {insights.data!.themes!.window_hours} h</p>
                {t.summary && <p className="text-sm text-slate-300 mt-2">{t.summary}</p>}
                {t.quotes.map((q, i) => <p key={i} className="text-xs italic text-slate-500 mt-1">“{q}”</p>)}
              </Card>
            ))}
          </div>
          {!insights.data.themes && (
            <Card><EmptyState title="No themes yet" hint="They need at least 20 posts in the last 3 days." /></Card>
          )}
          {insights.data.themes?.budget_exhausted && (
            <p className="text-xs text-amber-400">The daily LLM token budget ran out: some themes are labelled by their top words.</p>
          )}
        </>
      )}
    </div>
  )
}
