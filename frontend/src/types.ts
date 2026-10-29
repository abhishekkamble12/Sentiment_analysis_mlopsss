// Shared types mirroring the FastAPI response contracts (see app.py).

export type Sentiment = 'positive' | 'neutral' | 'negative'

export interface PredictionResult {
  prediction: Sentiment
  confidence: number | null
  latency_seconds: number
  latency_ms: number
  model_used: string
  decision_source: string
  inference_mode: string
  fallback_triggered: boolean
  explanation: string | null
  votes: Record<string, { label: Sentiment; confidence: number | null }>
  warnings: string[]
}

export interface Analytics {
  total_insights: number
  avg_latency_ms: number
  avg_confidence: number | null
  sentiment_index: number | null
  low_confidence_count: number
  sentiment_distribution: Record<Sentiment, number>
  trends: {
    categories: string[]
    positive: number[]
    neutral: number[]
    negative: number[]
  }
  top_keywords: string[]
  reports_considered: number
}

export interface RecentLog {
  id: number
  timestamp: string | null
  model_used: string
  prediction: Sentiment
  latency_ms: number
  confidence: number | null
  llm: boolean
}

export interface ModelInfo {
  pipeline_ready: boolean
  best_model_name: string | null
  fast_model_name: string | null
  inference_mode: string | null
  transformer_enabled: boolean
  label_mapping: Record<string, string>
  best_metrics: Record<string, number> | null
  run_id: string | null
}

export interface HistoryLog {
  id: number
  timestamp: string | null
  text: string | null
  prediction: Sentiment
  model_used: string
  confidence: number | null
  latency_ms: number
  text_length: number
  llm: boolean
}

export interface PredictionHistory {
  logs: HistoryLog[]
  total: number
  page: number
  limit: number
  total_pages: number
}

export interface CsvRowResult {
  text: string
  prediction: Sentiment
  confidence: number | null
  model_used: string
}

export interface CsvUploadResponse {
  results: CsvRowResult[]
  total_processed: number
  distribution: Record<Sentiment, number>
  errors: { text: string; error: string }[]
}

export interface PerClassMetric {
  precision: number
  recall: number
  f1: number
  support: number
}

export interface LabModel {
  model: string
  accuracy: number
  macro_f1: number
  weighted_f1: number
  kappa: number
  ece: number | null
  ci_low: number | null
  ci_high: number | null
  baseline: boolean
  per_class: Record<Sentiment, PerClassMetric> | null
  confusion_matrix: number[][] | null
  latency: { n: number; p50_ms: number; p95_ms: number; mean_ms: number } | null
}

export interface LabRun {
  run_id: string | null
  split: {
    n_test?: number
    label_source?: string
    seed?: number
    test_texts_also_in_train_pct?: number
    weights_column?: string | null
    [k: string]: unknown
  } | null
  models: LabModel[]
  comparisons: {
    model_a: string
    model_b: string
    macro_f1_diff: number
    ci_low: number
    ci_high: number
    [k: string]: unknown
  }[]
}

export interface ModelLab {
  silver: LabRun | null
  gold: LabRun | null
  label_order: Sentiment[]
  best_model_name: string | null
  limitation: string
}

export interface ApiError {
  detail: string
}
