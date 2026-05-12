import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const proxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000'
const proxyMs = Number.parseInt(process.env.VITE_PROXY_TIMEOUT_MS ?? `${15 * 60 * 1000}`, 10)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
        timeout: proxyMs,
        proxyTimeout: proxyMs,
      },
      '/health': {
        target: proxyTarget,
        changeOrigin: true,
        timeout: proxyMs,
        proxyTimeout: proxyMs,
      },
    },
  },
})
