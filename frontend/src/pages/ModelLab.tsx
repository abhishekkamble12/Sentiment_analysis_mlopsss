import { useState } from 'react'
import { FlaskConical, Info, Layers } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import { useAsync } from '../hooks/useAsync'
import type { LabModel, LabRun, ModelLab as ModelLabData, Sentiment } from '../types'
import { Card, EmptyState, ErrorBanner, Skeleton, fmtNum } from '../components/ui'
import { F1BarChart } from '../components/charts'

export default function ModelLab() {
  const lab = useAsync<ModelLabData>(() => api.modelLab())

  if (lab.loading && !lab.data) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-16" />
        <Skeleton className="h-64" />
      </div>
    )
  }
  if (lab.error) {
    return <ErrorBanner message={errorMessage(new Error(lab.error))} onRetry={lab.refetch} />
  }
  if (!lab.data) return <EmptyState title="No evaluation data" />

  return (
    <div className="space-y-6">
      {/* Limitation banner */}
      <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-sm">
        <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">Silver labels — not human annotations</p>
          <p className="text-amber-200/80 mt-0.5">{lab.data.limitation}</p>
        </div>
      </div>

      {lab.data.silver ? (
        <RunSection title="Silver Evaluation (TextBlob-generated labels)" run={lab.data.silver} best={lab.data.best_model_name} labelOrder={lab.data.label_order} />
      ) : (
        <Card><EmptyState title="No silver evaluation found" hint="Run python main.py to generate artifacts/evaluation/results.json" /></Card>
      )}

      {lab.data.gold ? (
        <RunSection title="Gold Evaluation (Human-labelled)" run={lab.data.gold} best={null} labelOrder={lab.data.label_order} />
      ) : (
        <Card>
          <div className="flex items-center gap-2 text-slate-400">
            <Layers className="w-4 h-4" />
            <span className="text-sm">Gold (human-labelled) evaluation is pending. It appears once the gold set is annotated and scored.</span>
          </div>
        </Card>
      )}
    </div>
  )
}

function RunSection({ title, run, best, labelOrder }: { title: string; run: LabRun; best: string | null; labelOrder: Sentiment[] }) {
  const split = run.split
  const trained = run.models.filter((m) => !m.baseline)
  const barData = run.models.map((m) => ({ model: m.model, macro_f1: m.macro_f1 ?? 0 })).sort((a, b) => b.macro_f1 - a.macro_f1)

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-white flex items-center gap-2">
          <FlaskConical className="w-5 h-5 text-indigo-400" /> {title}
        </h3>
        {split && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
            {split.n_test != null && <span>n_test: <span className="text-slate-200 font-mono">{split.n_test.toLocaleString()}</span></span>}
            {split.label_source && <span>labels: <span className="text-slate-200">{String(split.label_source)}</span></span>}
            {split.test_texts_also_in_train_pct != null && <span>train/test overlap: <span className="text-emerald-400 font-mono">{fmtNum(split.test_texts_also_in_train_pct, 1)}%</span></span>}
            {best && <span>serving: <span className="text-indigo-300 font-mono">{best}</span></span>}
          </div>
        )}
      </div>

      {/* F1 comparison chart */}
      <Card>
        <h4 className="text-sm font-semibold text-slate-300 mb-3">Macro-F1 comparison</h4>
        <F1BarChart data={barData} />
      </Card>

      {/* Metrics table */}
      <Card>
        <h4 className="text-sm font-semibold text-slate-300 mb-3">Model metrics</h4>
        <div className="overflow-x-auto -mx-2 px-2">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500 border-b border-slate-800">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 px-3 font-medium text-right">Accuracy</th>
                <th className="py-2 px-3 font-medium text-right">Macro-F1 [95% CI]</th>
                <th className="py-2 px-3 font-medium text-right">Weighted F1</th>
                <th className="py-2 px-3 font-medium text-right">Kappa</th>
                <th className="py-2 px-3 font-medium text-right">ECE</th>
                <th className="py-2 px-3 font-medium text-right">Latency p50</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {run.models.map((m) => (
                <tr key={m.model} className={m.model === best ? 'bg-indigo-500/5' : ''}>
                  <td className="py-2.5 pr-4">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-100 font-medium">{m.model}</span>
                      {m.model === best && <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300">serving</span>}
                      {m.baseline && <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400">baseline</span>}
                    </div>
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-300">{fmtNum(m.accuracy, 4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono">
                    <span className="text-slate-100">{fmtNum(m.macro_f1, 4)}</span>
                    {m.ci_low != null && m.ci_high != null && (
                      <span className="text-slate-500 text-xs"> [{fmtNum(m.ci_low, 4)}, {fmtNum(m.ci_high, 4)}]</span>
                    )}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-300">{fmtNum(m.weighted_f1, 4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-300">{fmtNum(m.kappa, 4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-300">{m.ece == null ? '—' : fmtNum(m.ece, 4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                    {m.latency ? `${fmtNum(m.latency.p50_ms, 2)} ms` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Per-model detail: per-class F1 + confusion matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {trained.map((m) => (
          <ModelDetail key={m.model} model={m} labelOrder={labelOrder} />
        ))}
      </div>

      {/* Pairwise comparisons */}
      {run.comparisons.length > 0 && (
        <Card>
          <h4 className="text-sm font-semibold text-slate-300 mb-3">Pairwise comparisons (paired bootstrap, Holm-adjusted)</h4>
          <div className="overflow-x-auto -mx-2 px-2">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-500 border-b border-slate-800">
                  <th className="py-2 pr-4 font-medium">Pair</th>
                  <th className="py-2 px-3 font-medium text-right">F1 diff</th>
                  <th className="py-2 px-3 font-medium text-right">95% CI</th>
                  <th className="py-2 px-3 font-medium text-right">p (bootstrap)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {run.comparisons.slice(0, 12).map((c, i) => (
                  <tr key={i}>
                    <td className="py-2 pr-4 text-slate-300">{c.model_a} vs {c.model_b}</td>
                    <td className="py-2 px-3 text-right font-mono text-slate-300">{fmtNum(c.macro_f1_diff, 4)}</td>
                    <td className="py-2 px-3 text-right font-mono text-slate-400">[{fmtNum(c.ci_low, 4)}, {fmtNum(c.ci_high, 4)}]</td>
                    <td className="py-2 px-3 text-right font-mono text-slate-400">{fmtNum((c as Record<string, unknown>).p_bootstrap as number, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

function ModelDetail({ model, labelOrder }: { model: LabModel; labelOrder: Sentiment[] }) {
  const [open, setOpen] = useState(false)
  const cm = model.confusion_matrix
  const maxCell = cm ? Math.max(...cm.flat(), 1) : 1

  return (
    <Card>
      <button onClick={() => setOpen((o) => !o)} className="w-full flex items-center justify-between text-left">
        <span className="text-sm font-semibold text-white">{model.model}</span>
        <span className="text-xs text-slate-400">{open ? 'Hide' : 'Per-class & confusion matrix'}</span>
      </button>

      {open && (
        <div className="mt-4 space-y-4">
          {/* Per-class F1 */}
          {model.per_class && (
            <div className="grid grid-cols-3 gap-2">
              {labelOrder.map((s) => {
                const c = model.per_class?.[s]
                if (!c) return null
                return (
                  <div key={s} className="glass-card p-2.5 text-center">
                    <p className="text-xs text-slate-400 capitalize">{s}</p>
                    <p className="text-lg font-bold text-white">{fmtNum(c.f1, 3)}</p>
                    <p className="text-[10px] text-slate-500">sup {c.support}</p>
                  </div>
                )
              })}
            </div>
          )}

          {/* Confusion matrix heatmap */}
          {cm && (
            <div className="overflow-x-auto">
              <table className="text-xs mx-auto">
                <thead>
                  <tr>
                    <th className="p-1.5"></th>
                    {labelOrder.map((s) => (
                      <th key={s} className="p-1.5 font-medium text-slate-400 capitalize">{s.slice(0, 3)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cm.map((row, i) => (
                    <tr key={i}>
                      <td className="p-1.5 font-medium text-slate-400 capitalize text-right pr-2">{labelOrder[i]?.slice(0, 3)}</td>
                      {row.map((cell, j) => {
                        const intensity = cell / maxCell
                        return (
                          <td key={j} className="p-1">
                            <div
                              className="w-14 h-9 flex items-center justify-center rounded font-mono text-slate-100"
                              style={{ background: `rgba(99, 102, 241, ${0.15 + intensity * 0.65})` }}
                              title={`true ${labelOrder[i]} → pred ${labelOrder[j]}`}
                            >
                              {cell.toLocaleString()}
                            </div>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-[10px] text-slate-500 text-center mt-1">rows = true label, columns = predicted</p>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}
