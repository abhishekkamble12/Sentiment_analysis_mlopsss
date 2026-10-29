import { useState } from 'react'
import { Sparkles, Send, Zap } from 'lucide-react'
import { api, errorMessage } from '../api/client'
import type { PredictionResult, Sentiment } from '../types'
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

  const submit = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
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
          <ResultView result={result} />
        ) : (
          <div className="flex-1 flex items-center justify-center text-sm text-slate-500 text-center px-6">
            Submit text to see the predicted sentiment, confidence, decision source and latency.
          </div>
        )}
      </Card>
    </div>
  )
}

function ResultView({ result }: { result: PredictionResult }) {
  const sentiment = normaliseSentiment(result.prediction) as Sentiment
  const meta = SENTIMENT_META[sentiment]
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
        {result.explanation ? (
          <p className="text-sm text-slate-300 whitespace-pre-wrap">{result.explanation}</p>
        ) : (
          <p className="text-sm text-slate-500 italic">
            No explanation — this prediction came from a local model ({result.decision_source}), which outputs a
            label and probability only. Explanations appear when an LLM decides (INFERENCE_MODE=llm/hybrid).
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
      </div>
    </div>
  )
}
