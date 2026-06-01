import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const appBase = process.env.VITE_BASE_PATH ?? '/ocr/'
const proxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8000'
const proxyMs = Number.parseInt(process.env.VITE_PROXY_TIMEOUT_MS ?? `${15 * 60 * 1000}`, 10)
const apiPrefix = `${appBase.replace(/\/$/, '')}/api`

// https://vite.dev/config/
export default defineConfig({
  base: appBase,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      [apiPrefix]: {
        target: proxyTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(new RegExp(`^${appBase.replace(/\/$/, '')}`), ''),
        timeout: proxyMs,
        proxyTimeout: proxyMs,
      },
    },
  },
})
