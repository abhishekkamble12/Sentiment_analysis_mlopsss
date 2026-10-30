import { useEffect, useState } from 'react'
import { Bot, Send } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import type { AgentRun } from '../types'
import { Card, ErrorBanner, PostBadge, Spinner } from '../components/ui'
import { useAsync } from '../hooks/useAsync'

const POLL_MS = 1500

export default function Agent() {
  const topics = useAsync(() => api.topics(), [])
  const [kind, setKind] = useState<'ask' | 'report'>('ask')
  const [topicId, setTopicId] = useState<number | ''>('')
  const [question, setQuestion] = useState('')
  const [runId, setRunId] = useState<number | null>(null)
  const [run, setRun] = useState<AgentRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Poll the session until it leaves "running"; stop when the page changes or a new session starts.
  useEffect(() => {
    if (runId == null) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const tick = async () => {
      try {
        const next = await api.agentRun(runId)
        if (cancelled) return
        setRun(next)
        if (next.status === 'running') timer = setTimeout(tick, POLL_MS)
        else setBusy(false)
      } catch (err) {
        if (cancelled) return
        setError(errorMessage(err, 'Lost contact with the session'))
        setBusy(false)
      }
    }
    tick()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [runId])

  const canStart = !busy && (kind === 'report' ? topicId !== '' : question.trim().length > 0)

  const start = async () => {
    setError(null)
    setRun(null)
    setBusy(true)
    try {
      const started = await api.startAgent({
        kind, question: question.trim() || undefined, topic_id: topicId === '' ? null : topicId,
      })
      setRunId(started.id)
    } catch (err) {
      setError(errorMessage(err, 'Could not start the session'))
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <p className="text-sm text-slate-400 mb-4">
          The agent answers from your stored posts using a fixed set of tools (search, trends, themes, alerts). It cites
          the posts it used, and any citation it cannot back with a retrieved post is removed. Starting a session spends
          LLM tokens, so it needs an API key (set it in the sidebar).
        </p>
        <div className="grid grid-cols-1 md:grid-cols-[12rem_1fr_auto] gap-3">
          <select className="input-base" value={kind} onChange={(e) => setKind(e.target.value as 'ask' | 'report')}>
            <option value="ask">Ask a question</option>
            <option value="report">Research report</option>
          </select>
          <select className="input-base" value={topicId}
            onChange={(e) => setTopicId(e.target.value === '' ? '' : Number(e.target.value))}>
            <option value="">{kind === 'report' ? 'Choose a topic' : 'Any topic'}</option>
            {topics.data?.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <button className="btn-primary" disabled={!canStart} onClick={start}>
            {busy ? <Spinner /> : <Send className="w-4 h-4" />} {busy ? 'Working…' : 'Run'}
          </button>
        </div>
        {kind === 'ask' && (
          <input className="input-base mt-3" maxLength={500} placeholder="e.g. Why did negativity rise this week?"
            value={question} onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && canStart && start()} />
        )}
        {error && <div className="mt-3"><ErrorBanner message={error} /></div>}
      </Card>

      {run && (
        <div className="grid grid-cols-1 xl:grid-cols-[22rem_1fr] gap-4">
          <Card>
            <h4 className="text-sm font-semibold text-white mb-2 flex items-center gap-2"><Bot className="w-4 h-4" /> What the agent did</h4>
            <ol className="space-y-2 text-xs">
              {run.steps.map((s) => (
                <li key={s.step}>
                  <p className="text-slate-200">
                    {s.step}. {s.action === 'final' ? 'wrote the answer' : `${s.tool} ${JSON.stringify(s.args)}`}
                  </p>
                  {s.result && <p className="text-slate-500 truncate" title={s.result}>{s.result}</p>}
                </li>
              ))}
              {run.status === 'running' && <li className="text-slate-500">working…</li>}
            </ol>
            <p className="text-xs text-slate-500 mt-3">
              {run.status} · {run.tokens_in + run.tokens_out} tokens
              {run.latency_ms != null && ` · ${(run.latency_ms / 1000).toFixed(1)} s`}
              {run.model && ` · ${run.provider}/${run.model}`}
            </p>
          </Card>

          <Card>
            {run.error && <ErrorBanner message={run.error} />}
            {run.answer && (
              <div>
                {/* model output built from scraped text: rendered as text, never as HTML */}
                <p className="text-sm text-slate-200 whitespace-pre-wrap">{run.answer.text}</p>
                {!run.answer.verified && (
                  <p className="mt-3 text-xs text-amber-400">
                    This answer cites no retrieved post, so it could not be checked against the data.
                  </p>
                )}
                {run.answer.removed_citations.length > 0 && (
                  <p className="mt-1 text-xs text-amber-400">
                    Removed citations with no matching retrieved post: {run.answer.removed_citations.join(', ')}
                  </p>
                )}
                {run.answer.sources.length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs text-slate-500 mb-1">Cited posts</p>
                    <div className="space-y-1.5">
                      {run.answer.sources.map((s) => (
                        <div key={s.id} className="p-2 rounded-lg border border-slate-800 text-xs">
                          <div className="flex items-center gap-2 mb-1"><span className="text-indigo-300">[{s.id}]</span><PostBadge label={s.sentiment} /></div>
                          <span className="text-slate-300">{s.text}</span>
                          {s.url && <> <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:underline">source</a></>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
