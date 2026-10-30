import axios from 'axios'
import type {
  AgentRun,
  Alert,
  Analytics,
  AnnotationItem,
  Drift,
  Insights,
  Post,
  PostsResponse,
  Topic,
  TrendPoint,
  CsvUploadResponse,
  ModelInfo,
  ModelLab,
  PredictionHistory,
  PredictionResult,
  RecentLog,
  Sentiment,
} from '../types'

const KEY_STORAGE = 'api-key'

/** The API key lives in sessionStorage (gone when the tab closes) and is sent as X-API-Key when present. */
export const keyStore = {
  get: (): string => {
    try {
      return sessionStorage.getItem(KEY_STORAGE) ?? ''
    } catch {
      return ''
    }
  },
  set: (key: string) => {
    try {
      if (key) sessionStorage.setItem(KEY_STORAGE, key)
      else sessionStorage.removeItem(KEY_STORAGE)
    } catch {
      /* storage blocked: the key then lasts until the page is reloaded */
    }
  },
}

// Relative paths work in both modes:
//  - dev: the Vite proxy forwards /api to the backend on :8000
//  - prod: the backend serves the built app at /app on the same origin, so /api is same-origin
const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? '',
  timeout: 60_000,
})

http.interceptors.request.use((config) => {
  const key = keyStore.get()
  if (key) config.headers.set('X-API-Key', key)
  return config
})

/** Read the FastAPI `detail` message from any error response. */
export function errorMessage(err: unknown, fallback = 'Something went wrong'): string {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data as { detail?: string } | undefined
    if (data?.detail) return data.detail
    if (err.message) return err.message
  }
  if (err instanceof Error) return err.message
  return fallback
}

export const api = {
  health: () => http.get<{ status: string; message: string }>('/health').then((r) => r.data),

  predict: (text: string) =>
    http.post<PredictionResult>('/api/predict', { text }).then((r) => r.data),

  submitCorrection: (body: {
    text: string
    predicted: Sentiment | 'uncertain'
    corrected: Sentiment
    decision_source: string
  }) => http.post<{ status: string; id: number }>('/api/v1/corrections', body).then((r) => r.data),

  uploadCsv: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return http
      .post<CsvUploadResponse>('/api/upload-csv', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((r) => r.data)
  },

  analytics: () => http.get<Analytics>('/api/analytics').then((r) => r.data),
  recentLogs: (limit = 8) =>
    http.get<{ logs: RecentLog[] }>('/api/recent-logs', { params: { limit } }).then((r) => r.data),
  modelInfo: () => http.get<ModelInfo>('/api/model-info').then((r) => r.data),
  modelLab: () => http.get<ModelLab>('/api/model-lab').then((r) => r.data),

  predictionHistory: (params: {
    q?: string
    sentiment?: string
    start?: string
    end?: string
    page?: number
    limit?: number
  }) => http.get<PredictionHistory>('/api/prediction-history', { params }).then((r) => r.data),

  // ---- topic monitoring ----
  topics: () => http.get<{ topics: Topic[] }>('/api/v1/topics').then((r) => r.data.topics),
  createTopic: (body: { name: string; query: string }) =>
    http.post<{ id: number }>('/api/v1/topics', body).then((r) => r.data),
  deleteTopic: (id: number) => http.delete(`/api/v1/topics/${id}`).then((r) => r.data),
  posts: (id: number, params: { sentiment?: string; page?: number; limit?: number }) =>
    http.get<PostsResponse>(`/api/v1/topics/${id}/posts`, { params }).then((r) => r.data),
  trends: (id: number, days = 14) =>
    http.get<{ points: TrendPoint[] }>(`/api/v1/topics/${id}/trends`, { params: { days } }).then((r) => r.data.points),
  drift: (id: number) => http.get<Drift>(`/api/v1/topics/${id}/drift`).then((r) => r.data),
  alerts: (id: number) => http.get<{ alerts: Alert[] }>(`/api/v1/topics/${id}/alerts`).then((r) => r.data.alerts),
  ackAlert: (id: number) => http.post<Alert>(`/api/v1/alerts/${id}/ack`).then((r) => r.data),
  insights: (id: number) => http.get<Insights>(`/api/v1/topics/${id}/insights`).then((r) => r.data),
  runInsights: (id: number) => http.post(`/api/v1/topics/${id}/insights/run`).then((r) => r.data),
  ingestTopic: (id: number) => http.post(`/api/v1/topics/${id}/ingest`).then((r) => r.data),

  // ---- agent ----
  startAgent: (body: { kind: 'report' | 'ask'; question?: string; topic_id?: number | null }) =>
    http.post<{ id: number; status: string }>('/api/v1/agent/runs', body).then((r) => r.data),
  agentRun: (id: number) => http.get<AgentRun>(`/api/v1/agent/runs/${id}`).then((r) => r.data),

  // ---- annotation queue ----
  annotationQueue: (topicId?: number) =>
    http.get<{ items: AnnotationItem[] }>('/api/v1/annotation/queue', { params: { topic_id: topicId, limit: 200 } })
      .then((r) => r.data.items),
  refreshQueue: (topicId: number) =>
    http.post<{ queued: number }>('/api/v1/annotation/refresh', null, { params: { topic_id: topicId } }).then((r) => r.data),
  exportQueue: (topicId?: number) =>
    http.get<unknown[]>('/api/v1/annotation/export', { params: { topic_id: topicId, limit: 500 } }).then((r) => r.data),
}
