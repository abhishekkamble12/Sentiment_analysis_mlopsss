import { useState } from 'react'
import { Sparkles, Send, Zap } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import type { Contributions, PredictionResult, Sentiment } from '../types'
import { normaliseSentiment, SENTIMENT_META, SentimentBadge, Card, Spinner, ConfidenceBar, fmtPct } from '../components/ui'

const EXAMPLES = [
  'I absolutely love the new policy, this is a great step forward!',
  'The service was okay, nothing special either way.',
  'This is the worst experience I have ever had. Terrible.',
]

export default function Predict() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [submitted, setSubmitted] = useState('')

  const submit = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setSubmitted(text.trim())
      setResult(await api.predict(text.trim()))
    } catch (err) {
      setError(errorMessage(err, 'Prediction failed'))
    } finally {
      setLoading(false)
    }
  }

  const disabled = loading || !text.trim()

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input */}
      <Card className="flex flex-col">
        <h3 className="text-base font-semibold text-white mb-3 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-indigo-400" /> Input Text
        </h3>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') submit()
          }}
          placeholder="Enter a tweet, review, or feedback text to classify…"
          className="input-base flex-1 min-h-[200px] resize-none"
        />
        <div className="flex items-center justify-between mt-3">
          <span className={`text-xs font-mono ${text.length > 1500 ? 'text-amber-400' : 'text-slate-500'}`}>
            {text.length} chars
          </span>
          <div className="flex gap-2">
            <button className="btn-ghost" onClick={() => setText(EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)])}>
              Try example
            </button>
            <button className="btn-primary" onClick={submit} disabled={disabled}>
              {loading ? <Spinner /> : <Send className="w-4 h-4" />}
              {loading ? 'Predicting…' : 'Predict Sentiment'}
            </button>
          </div>
        </div>
        <p className="text-xs text-slate-500 mt-2">Tip: Ctrl/⌘ + Enter to predict.</p>
        {error && <p className="mt-3 text-sm text-rose-400">{error}</p>}
      </Card>

      {/* Result */}
      <Card className="flex flex-col">
        <h3 className="text-base font-semibold text-white mb-3">Result</h3>
        {loading ? (
          <div className="flex-1 flex flex-col gap-4 justify-center">
            <div className="h-16 rounded-lg bg-slate-800/70 animate-pulse" />
            <div className="h-8 w-1/2 rounded-lg bg-slate-800/70 animate-pulse" />
            <div className="h-8 w-2/3 rounded-lg bg-slate-800/70 animate-pulse" />
          </div>
        ) : result ? (
          <ResultView result={result} text={submitted} />
        ) : (
          <div className="flex-1 flex items-center justify-center text-sm text-slate-500 text-center px-6">
            Submit text to see the predicted sentiment, confidence, decision source and latency.
          </div>
        )}
      </Card>
    </div>
  )
}

const UNCERTAIN_META = { label: 'Uncertain', text: 'text-amber-400', badge: 'border-amber-500/30', dot: 'bg-amber-400' }

function ResultView({ result, text }: { result: PredictionResult; text: string }) {
  const uncertain = result.prediction === 'uncertain'
  const sentiment = normaliseSentiment(result.prediction) as Sentiment
  const meta = uncertain ? UNCERTAIN_META : SENTIMENT_META[sentiment]
  const voteEntries = Object.entries(result.votes || {})

  return (
    <div className="flex flex-col gap-4 flex-1">
      {/* Sentiment + confidence */}
      <div className="grid grid-cols-2 gap-3">
        <div className={`glass-card p-4 border ${meta.badge.split(' ').filter((c) => c.startsWith('border')).join(' ')}`}>
          <p className="text-xs text-slate-400 mb-1">Predicted Sentiment</p>
          <div className="flex items-center gap-2">
            <span className={`text-2xl font-bold ${meta.text}`}>{meta.label}</span>
            <span className={`w-2.5 h-2.5 rounded-full ${meta.dot} animate-pulse`} />
          </div>
        </div>
        <div className="glass-card p-4">
          <ConfidenceBar confidence={result.confidence} />
        </div>
      </div>

      {/* Decision source + latency */}
      <div className="grid grid-cols-2 gap-3">
        <div className="glass-card p-4">
          <p className="text-xs text-slate-400 mb-1">Decision Source</p>
          <span className="inline-block px-2 py-1 text-xs font-mono rounded-md bg-slate-800 border border-slate-700 text-indigo-300">
            {result.decision_source}
          </span>
          <p className="text-xs text-slate-500 mt-2">
            Mode: <span className="text-slate-300">{result.inference_mode}</span>
            {result.fallback_triggered && <span className="text-rose-400"> · fallback used</span>}
          </p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-slate-400 mb-1">Inference Latency</p>
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-amber-400" />
            <span className="text-xl font-mono text-white">{result.latency_ms.toFixed(0)} ms</span>
          </div>
        </div>
      </div>

      {/* Explanation / votes */}
      <div className="glass-card p-4 flex-1 min-h-0 overflow-y-auto">
        <p className="text-xs text-slate-500 mb-2">Model reasoning & votes</p>
        {uncertain && (
          <p className="text-sm text-amber-300 mb-3">
            The model's confidence is below the abstain threshold, so no label is reported.
          </p>
        )}
        {result.explanation ? (
          <p className="text-sm text-slate-300 whitespace-pre-wrap">{result.explanation}</p>
        ) : result.contributions ? (
          <ContributionsView contributions={result.contributions} text={text} />
        ) : (
          <p className="text-sm text-slate-500 italic">
            No explanation is available for this model ({result.decision_source}).
          </p>
        )}

        {voteEntries.length > 0 && (
          <div className="mt-4 pt-3 border-t border-slate-800">
            <p className="text-xs text-slate-500 mb-2">Votes from each model that ran</p>
            <div className="space-y-1">
              {voteEntries.map(([source, vote]) => (
                <div key={source} className="flex justify-between text-xs">
                  <span className="text-slate-400 font-mono">{source}</span>
                  <span className="text-slate-200">
                    {vote.label}
                    {vote.confidence == null ? '' : ` (${fmtPct(vote.confidence, 0)})`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        {result.warnings?.length > 0 && (
          <p className="mt-3 text-xs text-amber-400">Warnings: {result.warnings.join(', ')}</p>
        )}
        <CorrectionControl result={result} text={text} />
      </div>
    </div>
  )
}

/** The submitted text with the terms that pushed toward (green) or against (red) the predicted label. */
function ContributionsView({ contributions, text }: { contributions: Contributions; text: string }) {
  // Each character takes the mark of the strongest term covering it (unigrams and bigrams overlap);
  // terms under 10% of the strongest weight are listed but not highlighted, so stopwords stay plain.
  const marks: ('for' | 'against' | null)[] = Array(text.length).fill(null)
  const all = [
    ...contributions.supporting.map((t) => ({ ...t, kind: 'for' as const })),
    ...contributions.opposing.map((t) => ({ ...t, kind: 'against' as const })),
  ].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
  const terms = all.filter((t) => Math.abs(t.weight) >= 0.1 * Math.abs(all[0]?.weight ?? 0))
  for (const t of terms)
    for (const [start, end] of t.spans)
      for (let i = start; i < end && i < text.length; i++) marks[i] ??= t.kind

  const segments: { text: string; kind: 'for' | 'against' | null }[] = []
  for (let i = 0; i < text.length; i++) {
    const last = segments[segments.length - 1]
    if (last && last.kind === marks[i]) last.text += text[i]
    else segments.push({ text: text[i], kind: marks[i] })
  }
  const label = SENTIMENT_META[normaliseSentiment(contributions.label)].label.toLowerCase()

  return (
    <div>
      <p className="text-sm text-slate-300 leading-relaxed">
        {segments.map((seg, i) =>
          seg.kind ? (
            <mark
              key={i}
              className={`rounded px-0.5 ${seg.kind === 'for' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}`}
            >
              {seg.text}
            </mark>
          ) : (
            <span key={i}>{seg.text}</span>
          ),
        )}
      </p>
      <div className="grid grid-cols-2 gap-3 mt-3 text-xs">
        <TermList title={`Toward ${label}`} terms={contributions.supporting} className="text-emerald-300" />
        <TermList title={`Against ${label}`} terms={contributions.opposing} className="text-rose-300" />
      </div>
      <p className="text-xs text-slate-500 mt-2">{contributions.note}</p>
    </div>
  )
}

function TermList({ title, terms, className }: { title: string; terms: Contributions['supporting']; className: string }) {
  return (
    <div>
      <p className="text-slate-500 mb-1">{title}</p>
      {terms.length === 0 ? (
        <p className="text-slate-600">none</p>
      ) : (
        <ul className="space-y-0.5">
          {terms.map((t) => (
            <li key={t.term} className="flex justify-between font-mono">
              <span className={className}>{t.term}</span>
              <span className="text-slate-500">{t.weight > 0 ? '+' : ''}{t.weight.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** "Wrong label?": stores a correction for review (never used for training on its own). */
function CorrectionControl({ result, text }: { result: PredictionResult; text: string }) {
  const [state, setState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const options = (['positive', 'neutral', 'negative'] as Sentiment[]).filter((s) => s !== result.prediction)

  const send = async (corrected: Sentiment) => {
    setState('saving')
    try {
      await api.submitCorrection({ text, predicted: result.prediction, corrected, decision_source: result.decision_source })
      setState('saved')
    } catch {
      setState('error')
    }
  }

  if (state === 'saved') return <p className="mt-3 text-xs text-emerald-400">Thanks, your correction was saved for review.</p>
  return (
    <div className="mt-3 pt-3 border-t border-slate-800 flex flex-wrap items-center gap-2 text-xs">
      <span className="text-slate-500">Wrong label? It should be:</span>
      {options.map((o) => (
        <button key={o} className="btn-ghost !py-1 !px-2 text-xs" disabled={state === 'saving'} onClick={() => send(o)}>
          {SENTIMENT_META[o].label}
        </button>
      ))}
      {state === 'error' && <span className="text-rose-400">Could not save, try again.</span>}
    </div>
  )
}
