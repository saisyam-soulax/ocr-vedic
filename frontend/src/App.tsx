import { Document, Packer, Paragraph, TextRun } from 'docx'
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'

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

type OcrResponse = {
  provider: string
  pages: OcrPage[]
  combined_text: string
}

const ACCEPT_EXT = '.pdf,application/pdf,image/*'

const DEFAULT_PROMPT = `You are an expert paleographer for Vedic and Sanskrit manuscripts.
Transcribe printed or handwritten Śruti/Smṛti text with maximal fidelity.
Preserve Devanāgarī conjuncts, daṇḍas, numerals, markers, and diacritic-rich Latin (IAST) if present.
Keep Udātta, Anudātta, Svarita, and kampas exactly as in the source. No commentary—plain text only.`

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

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

async function downloadDocx(filename: string, text: string) {
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

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

export default function App() {
  const mainInputId = useId()
  const modelOptionsDatalistId = useId()
  const modelInputId = useId()
  const providerSelectId = useId()
  const systemPromptId = useId()

  const mainInputRef = useRef<HTMLInputElement>(null)
  const providerRef = useRef('gemini')

  const [providers, setProviders] = useState<ProviderRow[]>([])
  const [provider, setProvider] = useState<string>('gemini')
  const [modelIdValue, setModelIdValue] = useState('')
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_PROMPT)
  const [fewShots, setFewShots] = useState<FewShotRow[]>([])
  const [mainFiles, setMainFiles] = useState<File[]>([])
  const [result, setResult] = useState<string | null>(null)
  const [meta, setMeta] = useState<OcrResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [copyDone, setCopyDone] = useState(false)

  useEffect(() => {
    providerRef.current = provider
  }, [provider])

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

  const activeProvider = useMemo(
    () => providers.find((p) => p.id === provider),
    [providers, provider],
  )

  const fewShotOk = fewShots.every(
    (r) =>
      (!r.file && !r.text.trim()) || (Boolean(r.file) && Boolean(r.text.trim())),
  )

  const canSubmit = mainFiles.length > 0 && !loading && fewShotOk

  const submit = async () => {
    setError(null)
    setResult(null)
    setMeta(null)
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

    const payloadShots = fewShots
      .filter((r) => r.file && r.text.trim())
      .map((r) => ({ expected_text: r.text.trim() }))

    const fd = new FormData()
    mainFiles.forEach((f) => fd.append('files', f))
    fd.append('provider', provider)
    const mid = modelIdValue.trim()
    if (mid) fd.append('model_id', mid)
    fd.append('system_prompt', systemPrompt)
    fd.append('few_shots', JSON.stringify(payloadShots))
    fewShots.filter((r) => r.file && r.text.trim()).forEach((r) => {
      if (r.file) fd.append('few_shot_files', r.file)
    })

    setLoading(true)
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
      const data = body as OcrResponse
      setMeta(data)
      setResult(data.combined_text)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'OCR request failed')
    } finally {
      setLoading(false)
    }
  }

  const copyResult = useCallback(async () => {
    if (!result) return
    try {
      await navigator.clipboard.writeText(result)
      setCopyDone(true)
      window.setTimeout(() => setCopyDone(false), 2000)
    } catch {
      setError('Could not copy to clipboard. Your browser may block clipboard access.')
    }
  }, [result])

  const downloadStamp = () => {
    const d = new Date()
    return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}-${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`
  }

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
                Production-grade multimodal OCR for Devanāgarī, dense diacritics, and Vedic svaras on
                difficult scans. Route to Gemini, Claude on Bedrock, or open vision models.
              </p>
            </div>
          </div>
          <span className="app-tag" title="Runs against your configured API credentials">
            Local · API
          </span>
        </div>
        <ol className="app-workflow" aria-label="Workflow">
          <li>
            <strong>1.</strong> Sources & model
          </li>
          <li>
            <strong>2.</strong> Instructions
          </li>
          <li>
            <strong>3.</strong> Optional few-shots
          </li>
          <li>
            <strong>4.</strong> Run & export
          </li>
        </ol>
      </header>

      <main id="main-content">
        <section className="card" aria-labelledby="section-sources">
          <div className="card__header">
            <h2 id="section-sources" className="card__title">
              Sources &amp; model
            </h2>
          </div>
          <div className="grid grid-two">
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
                placeholder={
                  activeProvider?.default_model_id ?? 'Server default when left blank'
                }
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
            </div>

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
            Optional. Upload small crops from the same edition and paste the exact gold transcription.
            Order is preserved; use arrows to reorder rows.
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

        <div
          className="toolbar toolbar--sticky"
          role="region"
          aria-label="Actions"
        >
          <button
            type="button"
            className="btn btn--primary"
            onClick={submit}
            disabled={!canSubmit}
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Transcribing…
              </>
            ) : (
              'Run OCR'
            )}
          </button>
          {result && (
            <>
              <button
                type="button"
                className="btn"
                onClick={() => copyResult()}
              >
                {copyDone ? 'Copied' : 'Copy text'}
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => downloadText(`vedic-ocr-${downloadStamp()}.txt`, result)}
              >
                Download .txt
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => void downloadDocx(`vedic-ocr-${downloadStamp()}.docx`, result)}
              >
                Download .docx
              </button>
            </>
          )}
        </div>

        <div aria-live="polite" aria-atomic="true" className="sr-only">
          {loading ? 'Transcription in progress.' : ''}
        </div>
        <div aria-live="assertive" aria-atomic="true">
          {error ? (
            <div className="alert" role="alert">
              <strong>Error</strong>
              {error}
            </div>
          ) : null}
        </div>

        {meta && result && (
          <section className="card" aria-labelledby="section-result">
            <div className="result-header">
              <h2 id="section-result" className="card__title" style={{ border: 'none', margin: 0 }}>
                Transcription
              </h2>
              <p className="result-meta">
                Provider <code>{meta.provider}</code> · {meta.pages.length} segment
                {meta.pages.length !== 1 ? 's' : ''}
              </p>
            </div>
            <div className="output" tabIndex={0}>
              {result}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}
