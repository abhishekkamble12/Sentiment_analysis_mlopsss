import axios from 'axios'
import type {
  Analytics,
  CsvUploadResponse,
  ModelInfo,
  ModelLab,
  PredictionHistory,
  PredictionResult,
  RecentLog,
  Sentiment,
} from '../types'

// Relative paths work in both modes:
//  - dev: the Vite proxy forwards /api to the backend on :8000
//  - prod: the backend serves the built app at /app on the same origin, so /api is same-origin
const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? '',
  timeout: 60_000,
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
}
