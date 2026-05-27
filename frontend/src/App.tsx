import { Document, Packer, Paragraph, TextRun } from 'docx'
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

type ProviderRow = {
  id: string
  label: string
  configured: boolean
  detail: string | null
  default_model_id: string | null
  model_options: string[]
}

type FewShotRow = {
  key: string
  file: File | null
  previewUrl: string | null
  text: string
}

type OcrPage = {
  index: number
  source_file: string
  page_in_source: number | null
  text: string
  mime_type: string | null
}

type OcrJobResponse = {
  job_id: string
  stream_url: string
}

type VllmState = 'stopped' | 'starting' | 'ready' | 'stopping' | 'error'

type VllmStatus = {
  state: VllmState
  reachable: boolean
  message: string | null
}

type SavedJob = {
  job_id: string
  ts: number       // epoch ms
  filename: string // first source filename
  pages: number
  provider: string
}

// ─── localStorage helpers ─────────────────────────────────────────────────────

const SAVED_JOBS_KEY = 'vedic-ocr:saved-jobs'
const MAX_SAVED = 20

function loadSavedJobs(): SavedJob[] {
  try {
    return JSON.parse(localStorage.getItem(SAVED_JOBS_KEY) ?? '[]') as SavedJob[]
  } catch {
    return []
  }
}

function upsertSavedJob(job: SavedJob): SavedJob[] {
  const prev = loadSavedJobs().filter((j) => j.job_id !== job.job_id)
  const next = [job, ...prev].slice(0, MAX_SAVED)
  localStorage.setItem(SAVED_JOBS_KEY, JSON.stringify(next))
  return next
}

function removeSavedJobById(job_id: string): SavedJob[] {
  const next = loadSavedJobs().filter((j) => j.job_id !== job_id)
  localStorage.setItem(SAVED_JOBS_KEY, JSON.stringify(next))
  return next
}

// ─── Utilities ────────────────────────────────────────────────────────────────

const ACCEPT_EXT = '.pdf,application/pdf,image/*'

function friendlyModelLabel(id: string): string {
  const lookup: Array<[RegExp, string]> = [
    [/^us\.anthropic\.claude-opus-4-7$/, 'Claude Opus 4.7 (inference profile, best quality)'],
    [/^anthropic\.claude-opus-4-7$/, 'Claude Opus 4.7 (best quality)'],
    [/^us\.anthropic\.claude-sonnet-4-6$/, 'Claude Sonnet 4.6 (inference profile, fast)'],
    [/^anthropic\.claude-sonnet-4-6$/, 'Claude Sonnet 4.6 (fast)'],
    [/^us\.anthropic\.claude-opus-4-/, 'Claude Opus 4 (inference profile)'],
    [/^anthropic\.claude-opus-4-/, 'Claude Opus 4'],
    [/^us\.anthropic\.claude-sonnet-4-/, 'Claude Sonnet 4 (inference profile)'],
    [/^anthropic\.claude-sonnet-4-/, 'Claude Sonnet 4'],
    [/^us\.anthropic\.claude-3-7-sonnet/, 'Claude 3.7 Sonnet (inference profile)'],
    [/^anthropic\.claude-3-5-sonnet-20241022/, 'Claude 3.5 Sonnet v2 (Oct 2024)'],
    [/^gemini-3\.1-pro-preview$/, 'Gemini 3.1 Pro (preview)'],
    [/^gemini-3\.1-pro$/, 'Gemini 3.1 Pro'],
    [/^gemini-2\.5-pro$/, 'Gemini 2.5 Pro (stable)'],
    [/^gemini-2\.5-flash$/, 'Gemini 2.5 Flash (fast & cheap)'],
    [/^gemini-2\.0-flash/, 'Gemini 2.0 Flash'],
    [/^meta\.llama3-2-90b/, 'Llama 3.2 90B Vision'],
    [/^meta\.llama3-2-11b/, 'Llama 3.2 11B Vision'],
    [/^us\.amazon\.nova-pro/, 'Amazon Nova Pro (inference profile)'],
    [/^amazon\.nova-pro/, 'Amazon Nova Pro'],
    [/^us\.amazon\.nova-lite/, 'Amazon Nova Lite (inference profile)'],
    [/^amazon\.nova-lite/, 'Amazon Nova Lite'],
  ]
  for (const [re, label] of lookup) {
    if (re.test(id)) return `${label} — ${id}`
  }
  return id
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

function downloadStamp(): string {
  const d = new Date()
  return (
    `${d.getFullYear()}` +
    `${String(d.getMonth() + 1).padStart(2, '0')}` +
    `${String(d.getDate()).padStart(2, '0')}-` +
    `${String(d.getHours()).padStart(2, '0')}` +
    `${String(d.getMinutes()).padStart(2, '0')}`
  )
}

async function downloadDocxBlob(filename: string, text: string) {
  const lines = text.split('\n')
  const doc = new Document({
    sections: [
      {
        properties: {},
        children: lines.map(
          (line) =>
            new Paragraph({
              children: [new TextRun({ text: line, font: 'Noto Serif' })],
            }),
        ),
      },
    ],
  })
  const blob = await Packer.toBlob(doc)
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function App() {
  const mainInputId = useId()
  const modelOptionsDatalistId = useId()
  const modelInputId = useId()
  const providerSelectId = useId()
  const systemPromptId = useId()

  const mainInputRef = useRef<HTMLInputElement>(null)
  // Track provider in a ref so SSE callbacks don't go stale
  const providerRef = useRef('gemini')
  // The active EventSource for the current job
  const esRef = useRef<EventSource | null>(null)

  // ── Provider / model ───────────────────────────────────────────────────────
  const [providers, setProviders] = useState<ProviderRow[]>([])
  const [provider, setProvider] = useState<string>('gemini')
  const [modelIdValue, setModelIdValue] = useState('')

  // ── System prompt ──────────────────────────────────────────────────────────
  const [systemPrompt, setSystemPrompt] = useState('')
  const defaultPromptFetched = useRef(false)

  // ── Few-shots ──────────────────────────────────────────────────────────────
  const [fewShots, setFewShots] = useState<FewShotRow[]>([])

  // ── Files ──────────────────────────────────────────────────────────────────
  const [mainFiles, setMainFiles] = useState<File[]>([])
  const [dragActive, setDragActive] = useState(false)

  // ── Job / streaming ────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(false)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [pages, setPages] = useState<OcrPage[]>([])
  const [pagesDone, setPagesDone] = useState(0)
  const [pagesTotal, setPagesTotal] = useState(0)
  const [streamDone, setStreamDone] = useState(false)
  const [elapsedSecs, setElapsedSecs] = useState<number | null>(null)

  // ── UI ─────────────────────────────────────────────────────────────────────
  const [error, setError] = useState<string | null>(null)
  const [copyDone, setCopyDone] = useState(false)
  const [savedJobs, setSavedJobs] = useState<SavedJob[]>(() => loadSavedJobs())

  // ── vLLM ───────────────────────────────────────────────────────────────────
  const [vllmStatus, setVllmStatus] = useState<VllmStatus | null>(null)
  const [vllmBusy, setVllmBusy] = useState(false)

  // ── Keep providerRef in sync ───────────────────────────────────────────────
  useEffect(() => {
    providerRef.current = provider
  }, [provider])

  // ── Fetch provider list ────────────────────────────────────────────────────
  useEffect(() => {
    fetch('/api/providers')
      .then((r) => r.json())
      .then((body: { providers: ProviderRow[] }) => {
        const list = body.providers ?? []
        setProviders(list)
        setModelIdValue((m) => {
          if (m !== '') return m
          const row = list.find((p) => p.id === providerRef.current)
          return row?.default_model_id ?? ''
        })
      })
      .catch(() => setProviders([]))
  }, [])

  // ── Fetch default system prompt ────────────────────────────────────────────
  useEffect(() => {
    if (defaultPromptFetched.current) return
    defaultPromptFetched.current = true
    fetch('/api/ocr/defaults')
      .then((r) => r.json())
      .then((body: { default_system_prompt?: string }) => {
        if (body.default_system_prompt) setSystemPrompt(body.default_system_prompt)
      })
      .catch(() => {
        setSystemPrompt(
          'You are an expert paleographer for Vedic and Sanskrit manuscripts.\n' +
          'Transcribe printed or handwritten Śruti/Smṛti text with maximal fidelity.\n' +
          'Preserve Devanāgarī conjuncts, daṇḍas, numerals, markers, and diacritic-rich Latin (IAST) if present.\n' +
          'Keep Udātta, Anudātta, Svarita, and kampas exactly as in the source. No commentary—plain text only.',
        )
      })
  }, [])

  // ── Poll vLLM status (only when provider = vllm_gemma) ────────────────────
  useEffect(() => {
    if (provider !== 'vllm_gemma') {
      setVllmStatus(null)
      return
    }
    let cancelled = false
    const poll = () => {
      fetch('/api/vllm/status')
        .then((r) => r.json())
        .then((s: VllmStatus) => { if (!cancelled) setVllmStatus(s) })
        .catch(() => { if (!cancelled) setVllmStatus(null) })
    }
    poll()
    const id = window.setInterval(poll, 3000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [provider])

  // ── vLLM actions ───────────────────────────────────────────────────────────
  const vllmLoad = useCallback(async () => {
    setVllmBusy(true)
    try { await fetch('/api/vllm/load', { method: 'POST' }) }
    finally { setVllmBusy(false) }
  }, [])

  const vllmUnload = useCallback(async () => {
    setVllmBusy(true)
    try { await fetch('/api/vllm/unload', { method: 'POST' }) }
    finally { setVllmBusy(false) }
  }, [])

  // ── Few-shot helpers ───────────────────────────────────────────────────────
  const addFewShot = useCallback(() => {
    setFewShots((rows) => [
      ...rows,
      { key: crypto.randomUUID(), file: null, previewUrl: null, text: '' },
    ])
  }, [])

  const moveFewShot = useCallback((idx: number, dir: -1 | 1) => {
    setFewShots((rows) => {
      const j = idx + dir
      if (j < 0 || j >= rows.length) return rows
      const next = [...rows]
      ;[next[idx], next[j]] = [next[j], next[idx]]
      return next
    })
  }, [])

  const removeFewShot = useCallback((key: string) => {
    setFewShots((rows) => {
      const row = rows.find((r) => r.key === key)
      if (row?.previewUrl) URL.revokeObjectURL(row.previewUrl)
      return rows.filter((r) => r.key !== key)
    })
  }, [])

  const onFewShotFile = useCallback((key: string, file: File | null) => {
    setFewShots((rows) =>
      rows.map((r) => {
        if (r.key !== key) return r
        if (r.previewUrl) URL.revokeObjectURL(r.previewUrl)
        if (!file) return { ...r, file: null, previewUrl: null }
        return { ...r, file, previewUrl: URL.createObjectURL(file) }
      }),
    )
  }, [])

  // ── File helpers ───────────────────────────────────────────────────────────
  const appendMainFiles = useCallback((files: File[]) => {
    if (!files.length) return
    const allowed = files.filter((f) => {
      const t = (f.type || '').toLowerCase()
      if (t.startsWith('image/') || t === 'application/pdf') return true
      return /\.pdf$/i.test(f.name)
    })
    if (allowed.length) setMainFiles((prev) => [...prev, ...allowed])
  }, [])

  const onMainInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const captured = Array.from(e.target.files ?? [])
      appendMainFiles(captured)
      e.target.value = ''
    },
    [appendMainFiles],
  )

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragActive(false)
      appendMainFiles(Array.from(e.dataTransfer.files))
    },
    [appendMainFiles],
  )

  const removeMainFile = useCallback((idx: number) => {
    setMainFiles((files) => files.filter((_, i) => i !== idx))
  }, [])

  // ── Derived ────────────────────────────────────────────────────────────────
  const activeProvider = useMemo(
    () => providers.find((p) => p.id === provider),
    [providers, provider],
  )

  const fewShotOk = fewShots.every(
    (r) => (!r.file && !r.text.trim()) || (Boolean(r.file) && Boolean(r.text.trim())),
  )

  const canSubmit = mainFiles.length > 0 && !loading && fewShotOk

  const combinedText = useMemo(
    () =>
      [...pages]
        .sort((a, b) => a.index - b.index)
        .map((p) => p.text)
        .join('\n\n'),
    [pages],
  )

  const hasResult = combinedText.length > 0

  // ── Cancel ─────────────────────────────────────────────────────────────────
  const handleCancel = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    setLoading(false)
  }, [])

  // ── Submit ─────────────────────────────────────────────────────────────────
  const submit = async () => {
    setError(null)
    setPages([])
    setPagesDone(0)
    setPagesTotal(0)
    setStreamDone(false)
    setElapsedSecs(null)
    setCurrentJobId(null)
    setCopyDone(false)

    if (!mainFiles.length) {
      setError('Add at least one PDF or image to transcribe.')
      return
    }
    const incomplete = fewShots.some(
      (r) => (r.file && !r.text.trim()) || (!r.file && !!r.text.trim()),
    )
    if (incomplete) {
      setError('Each few-shot row needs both a snippet image and its expected transcription.')
      return
    }

    const fd = new FormData()
    mainFiles.forEach((f) => fd.append('files', f))
    fd.append('provider', provider)
    const mid = modelIdValue.trim()
    if (mid) fd.append('model_id', mid)
    fd.append('system_prompt', systemPrompt)
    fd.append(
      'few_shots',
      JSON.stringify(
        fewShots
          .filter((r) => r.file && r.text.trim())
          .map((r) => ({ expected_text: r.text.trim() })),
      ),
    )
    fewShots.filter((r) => r.file && r.text.trim()).forEach((r) => {
      if (r.file) fd.append('few_shot_files', r.file)
    })

    setLoading(true)

    // Step 1: POST to enqueue the job
    let jobId: string
    try {
      const res = await fetch('/api/ocr', { method: 'POST', body: fd })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        const detail =
          typeof body?.detail === 'string'
            ? body.detail
            : Array.isArray(body?.detail)
              ? body.detail.map((x: { msg?: string }) => x.msg ?? JSON.stringify(x)).join('; ')
              : `Request failed (${res.status})`
        throw new Error(detail)
      }
      const data = body as OcrJobResponse
      jobId = data.job_id
      setCurrentJobId(jobId)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'OCR request failed')
      setLoading(false)
      return
    }

    // Step 2: open SSE stream
    // Capture snapshot of mutable values for use in callbacks
    const firstFilename = mainFiles[0]?.name ?? 'unknown'
    const submittedProvider = providerRef.current

    const es = new EventSource(`/api/ocr/${jobId}/stream`)
    esRef.current = es
    let isDone = false

    es.addEventListener('start', (e: MessageEvent) => {
      try {
        const d = JSON.parse(e.data) as { total: number }
        setPagesTotal(d.total)
      } catch { /* ignore parse errors */ }
    })

    es.addEventListener('page', (e: MessageEvent) => {
      try {
        const d = JSON.parse(e.data) as { data: OcrPage; done: number; total: number }
        setPages((prev) => {
          const filtered = prev.filter((p) => p.index !== d.data.index)
          return [...filtered, d.data]
        })
        setPagesDone(d.done)
        setPagesTotal(d.total)
      } catch { /* ignore */ }
    })

    es.addEventListener('page_error', (e: MessageEvent) => {
      try {
        const d = JSON.parse(e.data) as { detail: string; done: number; total: number }
        setPagesDone(d.done)
        setPagesTotal(d.total)
        setError((prev) => (prev ? `${prev}\n${d.detail}` : d.detail))
      } catch { /* ignore */ }
    })

    es.addEventListener('done', (e: MessageEvent) => {
      isDone = true
      let finalTotal = 0
      try {
        const d = JSON.parse(e.data) as { total: number; elapsed_seconds: number }
        setElapsedSecs(d.elapsed_seconds)
        setPagesDone(d.total)
        setPagesTotal(d.total)
        finalTotal = d.total
      } catch { /* ignore */ }
      es.close()
      esRef.current = null
      setStreamDone(true)
      setLoading(false)

      // Persist job to localStorage
      setSavedJobs(
        upsertSavedJob({
          job_id: jobId,
          ts: Date.now(),
          filename: firstFilename,
          pages: finalTotal,
          provider: submittedProvider,
        }),
      )
    })

    es.onerror = () => {
      // After 'done', the server closes the connection — browser fires onerror.
      // That's normal; ignore it if we already handled 'done'.
      if (isDone || es.readyState === EventSource.CLOSED) return
      setError('Stream connection lost. Any partial results are shown above.')
      es.close()
      esRef.current = null
      setLoading(false)
    }
  }

  // ── Copy ───────────────────────────────────────────────────────────────────
  const copyResult = useCallback(async () => {
    if (!combinedText) return
    try {
      await navigator.clipboard.writeText(combinedText)
      setCopyDone(true)
      window.setTimeout(() => setCopyDone(false), 2000)
    } catch {
      setError('Could not copy to clipboard. Your browser may block clipboard access.')
    }
  }, [combinedText])

  // ── Restore job from history ───────────────────────────────────────────────
  const restoreJob = useCallback(async (job_id: string) => {
    setError(null)
    setPages([])
    setPagesDone(0)
    setPagesTotal(0)
    setStreamDone(false)
    setElapsedSecs(null)
    setCurrentJobId(job_id)
    try {
      const r = await fetch(`/api/ocr/${job_id}/result`)
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const body = await r.json() as { pages?: OcrPage[] }
      const sorted = [...(body.pages ?? [])].sort((a, b) => a.index - b.index)
      setPages(sorted)
      setPagesTotal(sorted.length)
      setPagesDone(sorted.length)
      setStreamDone(true)
    } catch (e) {
      setError(
        `Could not load job ${job_id}: ${e instanceof Error ? e.message : 'unknown error'}`,
      )
    }
  }, [])

  // ── vLLM display helpers ───────────────────────────────────────────────────
  const vllmState: VllmState = vllmStatus?.state ?? 'stopped'

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <a href="#main-content" className="sr-only">
        Skip to main content
      </a>

      <header className="app-header">
        <div className="app-header__top">
          <div className="app-brand">
            <div className="app-brand__mark" aria-hidden="true">
              ॐ
            </div>
            <div className="app-brand__text">
              <h1>Vedic OCR Studio</h1>
              <p>
                Production-grade multimodal OCR for Devanāgarī, dense diacritics, and Vedic svaras
                on difficult scans. Route to Gemini, Claude on Bedrock, or local Gemma 4.
              </p>
            </div>
          </div>
          <span className="app-tag" title="Runs against your configured API credentials">
            Local · API
          </span>
        </div>
        <ol className="app-workflow" aria-label="Workflow">
          <li>
            <strong>1.</strong> Sources &amp; model
          </li>
          <li>
            <strong>2.</strong> Instructions
          </li>
          <li>
            <strong>3.</strong> Optional few-shots
          </li>
          <li>
            <strong>4.</strong> Run &amp; export
          </li>
        </ol>
      </header>

      <main id="main-content">
        {/* ── Sources & model ────────────────────────────────────────────── */}
        <section className="card" aria-labelledby="section-sources">
          <div className="card__header">
            <h2 id="section-sources" className="card__title">
              Sources &amp; model
            </h2>
          </div>
          <div className="grid grid-two">
            {/* Left column: provider + model + optional vLLM panel */}
            <div className="field">
              <label className="field-label" htmlFor={providerSelectId}>
                Provider
              </label>
              <select
                id={providerSelectId}
                value={provider}
                onChange={(e) => {
                  const next = e.target.value
                  setProvider(next)
                  providerRef.current = next
                  const row = providers.find((p) => p.id === next)
                  setModelIdValue(row?.default_model_id ?? '')
                }}
              >
                {providers.length ? (
                  providers.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                      {p.configured ? '' : ' (not configured)'}
                    </option>
                  ))
                ) : (
                  <>
                    <option value="gemini">Google Gemini</option>
                    <option value="bedrock_claude">AWS Bedrock — Claude</option>
                    <option value="bedrock_ocr">AWS Bedrock — Open multimodal</option>
                    <option value="vllm_gemma">Local — Gemma 4 (vLLM)</option>
                  </>
                )}
              </select>

              <div className="field-row">
                {activeProvider && (
                  <span
                    className={`badge ${activeProvider.configured ? 'badge--ok' : 'badge--bad'}`}
                    title={activeProvider.detail ?? undefined}
                  >
                    {activeProvider.configured ? 'Configured' : 'Needs credentials'}
                  </span>
                )}
              </div>

              <label className="field-label" htmlFor={modelInputId} style={{ marginTop: 14 }}>
                Model
              </label>
              <input
                id={modelInputId}
                type="text"
                list={modelOptionsDatalistId}
                value={modelIdValue}
                onChange={(e) => setModelIdValue(e.target.value)}
                placeholder={activeProvider?.default_model_id ?? 'Server default when left blank'}
                autoComplete="off"
                spellCheck={false}
                aria-describedby="model-hint"
              />
              <datalist id={modelOptionsDatalistId}>
                {(activeProvider?.model_options ?? []).map((opt) => (
                  <option key={opt} value={opt} label={friendlyModelLabel(opt)}>
                    {friendlyModelLabel(opt)}
                  </option>
                ))}
              </datalist>
              <p id="model-hint" className="field-note">
                Pick a suggestion or type any model ID your backend supports. Empty uses the server
                default for the selected provider.
              </p>

              {/* vLLM status panel — only shown when using the local model */}
              {provider === 'vllm_gemma' && (
                <div className="vllm-panel" aria-label="vLLM server status">
                  <div className="vllm-panel__row">
                    <span
                      className={`status-dot status-dot--${vllmState}`}
                      aria-hidden="true"
                    />
                    <span className="vllm-panel__state">
                      {vllmStatus
                        ? vllmState.charAt(0).toUpperCase() + vllmState.slice(1)
                        : 'Unknown'}
                    </span>
                    {vllmStatus?.message && (
                      <span className="vllm-panel__msg" title={vllmStatus.message}>
                        {vllmStatus.message}
                      </span>
                    )}
                    <button
                      type="button"
                      className="btn btn--sm"
                      onClick={() => void vllmLoad()}
                      disabled={
                        vllmBusy ||
                        vllmState === 'ready' ||
                        vllmState === 'starting'
                      }
                    >
                      {vllmState === 'starting' ? (
                        <>
                          <span className="spinner" aria-hidden="true" />
                          Loading…
                        </>
                      ) : (
                        'Load model'
                      )}
                    </button>
                    <button
                      type="button"
                      className="btn btn--sm btn--danger"
                      onClick={() => void vllmUnload()}
                      disabled={
                        vllmBusy ||
                        vllmState === 'stopped' ||
                        vllmState === 'stopping'
                      }
                    >
                      {vllmState === 'stopping' ? (
                        <>
                          <span className="spinner" aria-hidden="true" />
                          Stopping…
                        </>
                      ) : (
                        'Unload'
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Right column: file drop zone + queued file list */}
            <div className="field">
              <span className="field-label" id="batch-label">
                Documents
              </span>
              <div
                className={`file-drop ${dragActive ? 'file-drop--active' : ''}`}
                onDragEnter={(e) => {
                  e.preventDefault()
                  setDragActive(true)
                }}
                onDragOver={(e) => e.preventDefault()}
                onDragLeave={() => setDragActive(false)}
                onDrop={onDrop}
              >
                <input
                  ref={mainInputRef}
                  id={mainInputId}
                  type="file"
                  className="input-file-native"
                  accept={ACCEPT_EXT}
                  multiple
                  aria-labelledby="batch-label"
                  onChange={onMainInputChange}
                />
                <div className="file-drop__inner">
                  <p className="file-drop__title">Drop PDFs or images here</p>
                  <p className="file-drop__sub">
                    Or click to browse · PDF, PNG, JPEG, WebP · multiple files allowed
                  </p>
                </div>
              </div>
              <div className="field-row">
                <span className="badge">
                  {mainFiles.length} file{mainFiles.length !== 1 ? 's' : ''} queued
                </span>
              </div>
              <ul className="file-list" aria-label="Queued files">
                {mainFiles.map((f, i) => (
                  <li key={`${f.name}-${i}`} className="file-row">
                    <div className="file-row__meta">
                      <div className="file-row__name" title={f.name}>
                        {f.name}
                      </div>
                      <span className="badge" style={{ marginTop: 6 }}>
                        {formatBytes(f.size)}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="btn btn--danger btn--sm"
                      onClick={() => removeMainFile(i)}
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* ── System instructions ────────────────────────────────────────── */}
        <section className="card" aria-labelledby="section-prompt">
          <div className="card__header">
            <h2 id="section-prompt" className="card__title">
              System instructions
            </h2>
          </div>
          <div className="field">
            <label className="field-label" htmlFor={systemPromptId}>
              Prompt sent with every page
            </label>
            <textarea
              id={systemPromptId}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={8}
            />
          </div>
        </section>

        {/* ── Few-shot examples ──────────────────────────────────────────── */}
        <section className="card" aria-labelledby="section-fewshot">
          <div className="card__header">
            <h2 id="section-fewshot" className="card__title">
              Few-shot examples
            </h2>
            <button type="button" className="btn btn--sm" onClick={addFewShot}>
              + Add example
            </button>
          </div>
          <p className="card__hint">
            Optional. Upload small crops from the same edition and paste the exact gold
            transcription. Order is preserved; use arrows to reorder rows.
          </p>
          <div className="grid">
            {fewShots.map((row, idx) => (
              <div key={row.key} className="few-card">
                <div className="few-card__head">
                  <span className="few-card__title">Example {idx + 1}</span>
                  <div className="toolbar">
                    <button
                      type="button"
                      className="btn btn--sm"
                      onClick={() => moveFewShot(idx, -1)}
                      disabled={idx === 0}
                      aria-label="Move example up"
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      className="btn btn--sm"
                      onClick={() => moveFewShot(idx, 1)}
                      disabled={idx === fewShots.length - 1}
                      aria-label="Move example down"
                    >
                      ↓
                    </button>
                    <button
                      type="button"
                      className="btn btn--danger btn--sm"
                      onClick={() => removeFewShot(row.key)}
                    >
                      Remove
                    </button>
                  </div>
                </div>
                <div className="grid grid-two">
                  <div className="field">
                    <label className="field-label">Snippet image</label>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) =>
                        onFewShotFile(row.key, e.target.files?.item(0) ?? null)
                      }
                    />
                    {row.previewUrl && (
                      <img className="thumb" src={row.previewUrl} alt="" />
                    )}
                  </div>
                  <div className="field">
                    <label className="field-label">Expected text</label>
                    <textarea
                      value={row.text}
                      onChange={(e) =>
                        setFewShots((rows) =>
                          rows.map((r) =>
                            r.key === row.key ? { ...r, text: e.target.value } : r,
                          ),
                        )
                      }
                      rows={6}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Sticky action toolbar ──────────────────────────────────────── */}
        <div className="toolbar toolbar--sticky" role="region" aria-label="Actions">
          {loading ? (
            <>
              <button type="button" className="btn btn--danger" onClick={handleCancel}>
                Cancel
              </button>
              <span
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: '0.9rem',
                  color: 'var(--text-secondary)',
                }}
              >
                <span
                  className="spinner"
                  aria-hidden="true"
                  style={{
                    border: '2px solid var(--border)',
                    borderTopColor: 'var(--accent)',
                    marginRight: 0,
                  }}
                />
                {pagesTotal > 0 ? `${pagesDone} / ${pagesTotal} pages` : 'Starting…'}
              </span>
            </>
          ) : (
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => void submit()}
              disabled={!canSubmit}
            >
              Run OCR
            </button>
          )}

          {hasResult && !loading && (
            <>
              <button type="button" className="btn" onClick={() => void copyResult()}>
                {copyDone ? 'Copied ✓' : 'Copy text'}
              </button>
              {currentJobId ? (
                <>
                  <a
                    className="btn"
                    href={`/api/ocr/${currentJobId}/download.txt`}
                    download
                  >
                    Download .txt
                  </a>
                  <a
                    className="btn"
                    href={`/api/ocr/${currentJobId}/download.docx`}
                    download
                  >
                    Download .docx
                  </a>
                </>
              ) : (
                /* Fallback: client-side DOCX for restored jobs without a server-side file */
                <button
                  type="button"
                  className="btn"
                  onClick={() =>
                    void downloadDocxBlob(`vedic-ocr-${downloadStamp()}.docx`, combinedText)
                  }
                >
                  Download .docx
                </button>
              )}
            </>
          )}
        </div>

        {/* ── Progress bar ───────────────────────────────────────────────── */}
        {pagesTotal > 0 && (
          <div className="progress-strip">
            <div className="progress-strip__label">
              {streamDone
                ? `Complete — ${pagesDone} page${pagesDone !== 1 ? 's' : ''}${
                    elapsedSecs !== null ? ` in ${elapsedSecs.toFixed(1)} s` : ''
                  }`
                : `${pagesDone} of ${pagesTotal} pages transcribed`}
            </div>
            <div className="progress-strip__bar">
              <div
                className="progress-strip__fill"
                style={{ width: `${Math.round((pagesDone / pagesTotal) * 100)}%` }}
              />
            </div>
          </div>
        )}

        {/* ── Accessible live regions ────────────────────────────────────── */}
        <div aria-live="polite" aria-atomic="true" className="sr-only">
          {loading
            ? `Transcribing — ${pagesDone} of ${pagesTotal} pages done.`
            : streamDone
              ? 'Transcription complete.'
              : ''}
        </div>
        <div aria-live="assertive" aria-atomic="true">
          {error ? (
            <div className="alert" role="alert">
              <strong>Error</strong>
              {error}
            </div>
          ) : null}
        </div>

        {/* ── Transcription result ───────────────────────────────────────── */}
        {hasResult && (
          <section className="card" aria-labelledby="section-result">
            <div className="result-header">
              <h2
                id="section-result"
                className="card__title"
                style={{ border: 'none', margin: 0 }}
              >
                Transcription
              </h2>
              <p className="result-meta">
                {pages.length} page{pages.length !== 1 ? 's' : ''}
                {provider && (
                  <>
                    {' '}· <code>{provider}</code>
                  </>
                )}
                {elapsedSecs !== null && <> · {elapsedSecs.toFixed(1)} s</>}
              </p>
            </div>
            <div className="output" tabIndex={0}>
              {combinedText}
            </div>
          </section>
        )}

        {/* ── Recent jobs ────────────────────────────────────────────────── */}
        {savedJobs.length > 0 && (
          <section className="card" aria-labelledby="section-history">
            <div className="card__header">
              <h2 id="section-history" className="card__title">
                Recent jobs
              </h2>
            </div>
            <ul className="file-list" aria-label="Recent OCR jobs">
              {savedJobs.map((job) => (
                <li key={job.job_id} className="file-row">
                  <div className="file-row__meta">
                    <div className="file-row__name" title={job.job_id}>
                      {job.filename}
                    </div>
                    <span className="badge" style={{ marginTop: 6 }}>
                      {new Date(job.ts).toLocaleDateString()} · {job.pages}p · {job.provider}
                    </span>
                  </div>
                  <div className="toolbar">
                    <button
                      type="button"
                      className="btn btn--sm"
                      onClick={() => void restoreJob(job.job_id)}
                    >
                      Restore
                    </button>
                    <a
                      className="btn btn--sm"
                      href={`/api/ocr/${job.job_id}/download.txt`}
                      download
                    >
                      .txt
                    </a>
                    <a
                      className="btn btn--sm"
                      href={`/api/ocr/${job.job_id}/download.docx`}
                      download
                    >
                      .docx
                    </a>
                    <button
                      type="button"
                      className="btn btn--danger btn--sm"
                      onClick={() => setSavedJobs(removeSavedJobById(job.job_id))}
                      aria-label={`Remove job for ${job.filename}`}
                    >
                      ✕
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>
    </div>
  )
}
