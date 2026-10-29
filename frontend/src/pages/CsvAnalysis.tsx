import { useRef, useState } from 'react'
import { Upload, Download, FileSpreadsheet, AlertTriangle } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import type { CsvUploadResponse, Sentiment } from '../types'
import {
  Card, EmptyState, ErrorBanner, SentimentBadge, Skeleton, Spinner, normaliseSentiment,
} from '../components/ui'
import { SentimentDonut } from '../components/charts'

export default function CsvAnalysis() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [fileName, setFileName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<CsvUploadResponse | null>(null)

  const onFile = async (file: File) => {
    setFileName(file.name)
    setLoading(true)
    setError(null)
    setData(null)
    try {
      setData(await api.uploadCsv(file))
    } catch (err) {
      setError(errorMessage(err, 'Upload failed'))
    } finally {
      setLoading(false)
    }
  }

  const download = () => {
    if (!data) return
    const header = 'text,prediction,confidence,model_used\n'
    const rows = data.results
      .map((r) => {
        const safe = r.text.replace(/"/g, '""').replace(/\n/g, ' ')
        const conf = r.confidence == null ? '' : r.confidence
        return `"${safe}",${r.prediction},${conf},${r.model_used}`
      })
      .join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `predictions_${fileName.replace(/\.csv$/i, '')}_results.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      {/* Upload zone */}
      <Card>
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const f = e.dataTransfer.files?.[0]
            if (f) onFile(f)
          }}
          className="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-2xl p-8 text-center cursor-pointer transition-colors"
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
          />
          {loading ? (
            <div className="flex flex-col items-center gap-2 text-slate-300">
              <Spinner className="w-8 h-8 text-indigo-400" />
              <p>Processing {fileName}…</p>
            </div>
          ) : (
            <>
              <Upload className="w-10 h-10 text-indigo-400 mx-auto mb-3" />
              <p className="text-slate-200 font-medium">
                Drop a CSV here, or <span className="text-indigo-400 underline">browse</span>
              </p>
              <p className="text-sm text-slate-500 mt-1">
                Must contain a <code className="text-slate-300">text</code> column. Up to 1,000 rows in local mode.
              </p>
            </>
          )}
        </div>
        {error && <div className="mt-3"><ErrorBanner message={error} /></div>}
      </Card>

      {loading && (
        <Card>
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-10" />)}
          </div>
        </Card>
      )}

      {data && !loading && (
        <>
          {/* Summary */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <Card className="lg:col-span-1">
              <h3 className="text-sm font-semibold text-slate-300 mb-2">Distribution</h3>
              <SentimentDonut data={data.distribution as Record<Sentiment, number>} />
            </Card>
            <Card className="lg:col-span-2">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4 text-indigo-400" /> Summary
                </h3>
                <button className="btn-ghost" onClick={download}>
                  <Download className="w-4 h-4" /> Download CSV
                </button>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Stat label="Rows processed" value={data.total_processed} />
                <Stat label="Positive" value={data.distribution.positive} accent="text-emerald-400" />
                <Stat label="Neutral" value={data.distribution.neutral} accent="text-slate-300" />
                <Stat label="Negative" value={data.distribution.negative} accent="text-rose-400" />
              </div>
              {data.errors.length > 0 && (
                <div className="mt-4 flex items-start gap-2 text-sm text-amber-400">
                  <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  <span>{data.errors.length} row(s) failed to process and were skipped.</span>
                </div>
              )}
            </Card>
          </div>

          {/* Results table */}
          <Card>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Results</h3>
            {data.results.length === 0 ? (
              <EmptyState title="No rows processed" hint="The CSV had no non-empty text rows." />
            ) : (
              <div className="overflow-x-auto -mx-2 px-2">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-slate-500 border-b border-slate-800">
                      <th className="py-2 pr-4 font-medium">Text</th>
                      <th className="py-2 px-4 font-medium">Sentiment</th>
                      <th className="py-2 px-4 font-medium">Confidence</th>
                      <th className="py-2 px-4 font-medium">Model</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {data.results.slice(0, 200).map((r, i) => (
                      <tr key={i}>
                        <td className="py-2.5 pr-4 max-w-md">
                          <p className="text-slate-200 truncate">{r.text}</p>
                        </td>
                        <td className="py-2.5 px-4"><SentimentBadge sentiment={normaliseSentiment(r.prediction)} /></td>
                        <td className="py-2.5 px-4 font-mono text-slate-300">
                          {r.confidence == null ? '—' : `${Math.round(r.confidence * 100)}%`}
                        </td>
                        <td className="py-2.5 px-4 font-mono text-xs text-slate-400">{r.model_used}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {data.results.length > 200 && (
                  <p className="text-xs text-slate-500 mt-3">
                    Showing first 200 of {data.results.length}. Download the CSV for all results.
                  </p>
                )}
              </div>
            )}
          </Card>
        </>
      )}

      {!data && !loading && !error && (
        <Card>
          <EmptyState
            title="Upload a CSV to begin"
            hint="The file needs a 'text' column. Each row is classified and the results, distribution and errors are shown below."
          />
        </Card>
      )}

      {/* tiny inline stat used above */}
    </div>
  )
}

function Stat({ label, value, accent = 'text-white' }: { label: string; value: number; accent?: string }) {
  return (
    <div className="glass-card p-3">
      <p className="text-xs text-slate-400">{label}</p>
      <p className={`text-xl font-bold ${accent}`}>{value}</p>
    </div>
  )
}
