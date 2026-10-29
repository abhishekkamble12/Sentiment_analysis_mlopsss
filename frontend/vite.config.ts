import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the Vite server proxies API calls to the FastAPI backend on :8000.
// Prod: `npm run build` outputs to dist/, which the backend serves at /app (base matches).
const apiTarget = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  base: '/app/',
  server: {
    port: 5173,
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
      '/predict': { target: apiTarget, changeOrigin: true },
      '/upload_csv': { target: apiTarget, changeOrigin: true },
      '/monitoring': { target: apiTarget, changeOrigin: true },
      '/health': { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
