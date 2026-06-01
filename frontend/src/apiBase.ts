/** App is served under Vite `base` (default `/ocr/`). API is proxied at `/ocr/api/`. */
const BASE = import.meta.env.BASE_URL

export function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path.slice(1) : path
  return `${BASE}api/${p}`
}
