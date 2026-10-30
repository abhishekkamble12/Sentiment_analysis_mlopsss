import { useState } from 'react'
import { Download, RefreshCw } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import { Card, EmptyState, ErrorBanner, Skeleton, Spinner } from '../components/ui'
import { useAsync } from '../hooks/useAsync'

export default function Annotate() {
  const topics = useAsync(() => api.topics(), [])
  const [topicId, setTopicId] = useState<number | ''>('')
  const queue = useAsync(() => api.annotationQueue(topicId === '' ? undefined : topicId), [topicId])
  const [busy, setBusy] = useState<'refresh' | 'export' | null>(null)
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null)

  const refresh = async () => {
    if (topicId === '') return
    setBusy('refresh')
    setNotice(null)
    try {
      const { queued } = await api.refreshQueue(topicId)
      setNotice({ ok: true, text: queued ? `Queued ${queued} more posts.` : 'Nothing new to queue.' })
      queue.refetch()
    } catch (err) {
      setNotice({ ok: false, text: errorMessage(err, 'Could not refresh the queue') })
    } finally {
      setBusy(null)
    }
  }

  const download = async () => {
    setBusy('export')
    setNotice(null)
    try {
      const tasks = await api.exportQueue(topicId === '' ? undefined : topicId)
      const url = URL.createObjectURL(new Blob([JSON.stringify(tasks, null, 2)], { type: 'application/json' }))
      const link = document.createElement('a')
      link.href = url
      link.download = 'label-studio-tasks.json'
      link.click()
      URL.revokeObjectURL(url)
      setNotice({ ok: true, text: `Exported ${tasks.length} tasks.` })
    } catch (err) {
      setNotice({ ok: false, text: errorMessage(err, 'Export failed') })
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <p className="text-sm text-slate-400 mb-4">
          Posts the model was unsure about, most uncertain first. Export them to Label Studio (config in
          <code className="mx-1">docs/label_studio_config.xml</code>) so people can label them. The export carries the
          text only, with no model label, so annotators are not anchored. These are candidates for new annotation
          rounds: never add them to the frozen gold test set.
        </p>
        <div className="flex flex-wrap gap-3">
          <select className="input-base !w-auto" value={topicId}
            onChange={(e) => setTopicId(e.target.value === '' ? '' : Number(e.target.value))}>
            <option value="">All topics</option>
            {topics.data?.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <button className="btn-ghost" disabled={topicId === '' || busy !== null} onClick={refresh}
            title={topicId === '' ? 'Pick a topic to add its unsure posts to the queue' : undefined}>
            {busy === 'refresh' ? <Spinner className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />} Add unsure posts
          </button>
          <button className="btn-primary" disabled={busy !== null || !queue.data?.length} onClick={download}>
            {busy === 'export' ? <Spinner /> : <Download className="w-4 h-4" />} Label Studio JSON
          </button>
        </div>
        {notice && <p className={`mt-3 text-sm ${notice.ok ? 'text-emerald-400' : 'text-rose-400'}`}>{notice.text}</p>}
      </Card>

      <Card>
        {queue.loading && <Skeleton className="h-40" />}
        {queue.error && <ErrorBanner message={queue.error} onRetry={queue.refetch} />}
        {queue.data?.length === 0 && (
          <EmptyState title="The queue is empty" hint="Pick a topic and add its unsure posts, or let the worker do it." />
        )}
        <div className="space-y-2">
          {queue.data?.map((item) => (
            <div key={item.id} className="p-3 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className={item.reason === 'uncertain' ? 'text-amber-300' : 'text-slate-400'}>
                  {item.reason === 'uncertain' ? 'model abstained' : 'low confidence'}
                </span>
                <span className="text-slate-500">priority {item.priority.toFixed(2)}</span>
              </div>
              <p className="text-sm text-slate-300">{item.text}</p>
              {item.url && <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-xs text-indigo-400 hover:underline">source</a>}
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
