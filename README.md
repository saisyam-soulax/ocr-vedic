# Vedic OCR Studio

Production-oriented **multimodal OCR** for Vedic and Sanskrit sources: **Devanāgarī**, heavy **IAST diacritics**, **svaras**, and noisy scans. The stack is a **FastAPI** backend with pluggable **Google Gemini** and **AWS Bedrock** (Claude + an open multimodal model) plus a **React + Vite + TypeScript** UI.

## Features

- **Providers**: `gemini`, `bedrock_claude` (Converse API), `bedrock_ocr` (Converse-compatible vision model such as Llama 3.2 Vision or Amazon Nova—verify in your region), `vllm_dots` (local **[dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)** via [vLLM](https://docs.vllm.ai/) on NVIDIA GPU).
- **Model selection**: The UI reads `GET /api/providers` for each provider's default model id and optional comma-separated env allowlists; OCR accepts optional `model_id` to override the server default without redeploy.
- **Few-shot steering**: ordered example images + gold text sent before each page (Gemini, Bedrock, and vLLM).
- **PDFs**: rasterized server-side with **PyMuPDF** (per-page OCR, JSON + combined export).
- **API**: `POST /api/ocr`, `GET /api/providers`, `GET /health`, CORS-ready for local dev.

## Requirements

- **Python 3.11+** (backend)
- **Node.js 20+** (frontend)
- Credentials for at least one provider (see `.env.example`)

## Quick start (local)

### Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set at least the variables for providers you enable. For AWS Bedrock, you can omit access keys from `.env` entirely if credentials are already on your machine via **`aws configure`** (writes `~/.aws/credentials`; use `AWS_PROFILE` or the default profile) or via **`export AWS_ACCESS_KEY_ID=...`** **`AWS_SECRET_ACCESS_KEY=...`** and optional **`AWS_SESSION_TOKEN`** in your shell—the SDK reads those automatically. Combining `.env` (region, profiles, example model IDs) with shell-managed secrets is supported.

Start the API (with env loaded if using `.env`):

```bash
export $(grep -v '^#' ../.env | xargs)   # optional: load vars from repo .env into the shell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run from `backend/` so `app` imports resolve (or set `PYTHONPATH`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/ocr/`. The Vite dev server **proxies** `/ocr/api` to `http://127.0.0.1:8000` (override with `VITE_API_PROXY_TARGET`). Change the mount path with `VITE_BASE_PATH` (default `/ocr/`).

### Docker Compose (LAN / production UI)

From the repo root (after copying `.env.example` → `.env` and filling values):

```bash
docker compose up -d --build
```

- UI + API (single origin): `http://<this-machine-LAN-IP>/ocr/` (nginx on port **80** proxies `/ocr/api` to the backend; `http://<IP>/` redirects to `/ocr/`)
- Set `CORS_ORIGINS` in `.env` to include your LAN IP if browsers call the API from another origin

One-time host setup (firewall + start on boot):

```bash
./deploy/setup-host.sh
```

For local frontend dev only (port 5173), run `npm run dev` in `frontend/` with the backend on port 8000.

### Local GPU (vLLM + dots.ocr)

On a machine with an NVIDIA GPU (e.g. **DGX Spark GB10**), run the vLLM sidecar with the production stack:

```bash
# In .env set VLLM_ENABLED=true and VLLM_ON_DEMAND=true (see .env.example)
docker compose --profile vllm up -d --build
```

- First start downloads **[rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)** (~3B) into `~/.cache/huggingface`; vLLM health typically takes a few minutes.
- The UI shows **Local — dots.ocr (vLLM)** when `GET /api/providers` reports `vllm_dots` as configured.
- Use **Load model** / **Unload** in the UI to start or stop the GPU container on demand (`VLLM_ON_DEMAND=true`).
- Compose uses `vllm/vllm-openai:v0.11.0` (dots.ocr integrated upstream), single-GPU, `bfloat16`, prefix caching, and batch concurrency for multi-page PDFs.
- **Shared GPU (default):** `VLLM_GPU_MEMORY_UTILIZATION=auto` reads free VRAM from `nvidia-smi` at container start (no need to stop other jobs). Also uses `VLLM_CPU_OFFLOAD_GB=16` and a 4096 context. Rebuild the vLLM image after pulling: `docker compose --profile vllm build vllm`. If startup still fails, set a fixed lower value (e.g. `0.03`) or use Gemini/Bedrock while the GPU is full.
- **DGX Spark GB10:** Official `vllm-openai:v0.11.0` lacks FlashInfer kernels for sm_12.x. Compose sets `VLLM_USE_FLASHINFER_SAMPLER=0` (fixes `TopKMaskLogits … no kernel image`). PyTorch may still warn that sm_12.1 > max 12.0 — usually harmless once the sampler workaround is applied.
- **Dedicated GPU:** set `VLLM_GPU_MEMORY_UTILIZATION=0.45`, `VLLM_MAX_MODEL_LEN=16384`, `OCR_PAGE_CONCURRENCY=8` in `.env`.
- vLLM uses the GPU; cloud providers (Gemini/Bedrock) still work over the network from the same backend.
- LAN URL unchanged: `http://<this-machine-LAN-IP>/`

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` | Gemini API key. AI Studio keys start with `AIzaSy...`; Vertex AI / Agentic Platform keys start with `AQ....` (also set `GEMINI_USE_VERTEXAI=true` for those). |
| `GEMINI_MODEL` | Vision model id (default `gemini-3.1-pro-preview`). Override in `.env` when Google ships newer tiers. |
| `GEMINI_MODEL_OPTIONS` | Optional comma-separated list of Gemini model ids surfaced in `/api/providers` (falls back to `GEMINI_MODEL` alone if unset). |
| `GEMINI_USE_VERTEXAI` | `true` routes the SDK through Vertex AI (required for Agentic Platform keys and Vertex-only models such as `gemini-3.1-pro`). Also accepts `GOOGLE_GENAI_USE_VERTEXAI`. |
| `GOOGLE_CLOUD_PROJECT` | GCP project id (numeric or name) used when `GEMINI_USE_VERTEXAI=true`. |
| `GOOGLE_CLOUD_LOCATION` | GCP region for Vertex AI (default `us-central1`). |
| `AWS_REGION` | Bedrock region |
| `AWS_PROFILE` | Optional shared credentials profile |
| `BEDROCK_CLAUDE_MODEL_ID` | Claude model identifier enabled in your account |
| `BEDROCK_CLAUDE_MODEL_OPTIONS` | Optional comma-separated allowlist surfaced in `/api/providers` |
| `BEDROCK_OCR_MODEL_ID` | Multimodal non-Claude model id (must support **Converse** with images) |
| `BEDROCK_OCR_MODEL_OPTIONS` | Optional comma-separated allowlist for `bedrock_ocr` |
| `VLLM_ENABLED` | `true` to enable local dots.ocr via vLLM |
| `VLLM_BASE_URL` | OpenAI-compatible API base (Compose: `http://vllm:8000/v1`) |
| `VLLM_MODEL` | Model id served by vLLM (Compose default: `model` via `--served-model-name`) |
| `VLLM_MODEL_OPTIONS` | Optional comma-separated allowlist for `vllm_dots` |
| `VLLM_ON_DEMAND` | `true` lets the UI start/stop the vLLM Docker container |
| `VLLM_CONTAINER_NAME` | Docker container name (default `ocr-vedic-vllm-1`) |
| `VLLM_REQUEST_TIMEOUT_SECONDS` | HTTP timeout for local inference (default `600`) |
| `HF_TOKEN` | Optional Hugging Face token for model download in the vLLM container |
| `OCR_REQUEST_TIMEOUT_SECONDS` | Provider HTTP/read timeout (default `120`) |
| `CORS_ORIGINS` | Comma-separated browser origins |
| `UPLOAD_STORAGE_DIR` | OCR batches as `<dir>/<uuid>/` + `metadata.json`. Default: `<backend-root>/data/uploads` (gitignored) |
| `UPLOAD_RETAIN` | If `true`, keep batch folders after a successful OCR; if `false` (default), delete the batch folder on success |
| `UPLOAD_RETAIN_HOURS` | When `> 0`, periodically delete batch folders older than this many hours (by directory `mtime`; cleans failed runs too) |
| `VITE_API_PROXY_TARGET` | Vite proxy target (Compose: `http://backend:8000`) |
| `VITE_PROXY_TIMEOUT_MS` | Dev proxy socket timeout for `/api` and `/health` (default `900000` ms = 15 min; raise for large PDFs / slow models) |

### Gemini: AI Studio vs Vertex AI

The bundled `google-genai` SDK can talk to two different Google backends. Pick one based on the key you have:

- **Public AI Studio key** (`AIzaSy...`): set `GOOGLE_API_KEY=AIzaSy...` and leave `GEMINI_USE_VERTEXAI` unset (or `false`). Available models are the public ones — typically `gemini-2.5-pro` / `gemini-2.5-flash` and the `2.0` family.
- **Vertex AI / Agentic Platform key** (`AQ....`): set `GEMINI_USE_VERTEXAI=true` and `GOOGLE_API_KEY=<vertex-key>`. This uses **Vertex AI Express Mode** — project/region are implicit from the key, so `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are ignored by the SDK in this mode (the SDK rejects passing both an API key and a project/location).
- **Vertex AI with ADC** (no API key): set `GEMINI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT=<id>`, `GOOGLE_CLOUD_LOCATION=<region>` (default `us-central1`), leave `GOOGLE_API_KEY` empty, and authenticate with `gcloud auth application-default login`.

Vertex AI exposes additional model ids that are not on the public API (for example `gemini-3.1-pro`). If `GEMINI_USE_VERTEXAI=true` is set with neither an API key nor a project+location, the provider fails fast at startup with an explicit error.

## AWS / Bedrock IAM (summary)

Attach a policy allowing `bedrock:InvokeModel` / `bedrock:InvokeModelWithResponseStream` (and for Converse, **`bedrock:InvokeModel`** on the chosen model ARNs or `foundation-model/*` as appropriate for your org). Use **least privilege** to specific model IDs. Ensure the model is **enabled in the Bedrock console** for the account/region.

Replace the **example** IDs in `.env.example` with ones that are actually entitled in your account (Claude IDs and Llama/Nova IDs vary by region).

## API: `GET /api/providers`

JSON list of `{ id, label, configured, default_model_id?, model_options[] }` describing each OCR backend. Populate `GEMINI_MODEL_OPTIONS`, `BEDROCK_CLAUDE_MODEL_OPTIONS`, and `BEDROCK_OCR_MODEL_OPTIONS` with comma-separated model ids so the Studio UI can autocomplete them; omit those vars to advertise only each provider's single configured default (`GEMINI_MODEL`, `BEDROCK_*_MODEL_ID`).

## API: `POST /api/ocr`

`multipart/form-data`:

| Field | Description |
|-------|-------------|
| `files` | One or more PDFs and/or images (PNG, JPEG, WebP, GIF) |
| `provider` | `gemini` \| `bedrock_claude` \| `bedrock_ocr` \| `vllm_dots` |
| `model_id` | Optional non-empty override for the multimodal model id (otherwise each provider uses its configured default env) |
| `system_prompt` | Optional; strong default is applied if omitted |
| `few_shots` | JSON array: `[{"expected_text":"..." , "image_base64?":"..." , "mime_type?":"..."}]` |
| `few_shot_files[]` | Images for entries **without** `image_base64`, in **order** |

Response: `combined_text` plus `pages[]` with `source_file`, `page_in_source`, and `text`.

Unified Python helper (single image):

```python
from app.providers.base import transcribe_image

text = transcribe_image(
    image_bytes,
    "image/png",
    system_prompt,
    few_shot_examples,  # list[dict] with keys image_base64, expected_text, mime_type?
    "gemini",
    model_id="gemini-2.5-flash",  # optional
)
```

## Vedic OCR tuning tips

- **DPI**: PDFs render at **200 DPI** by default (`app/utils/pdf.py`); raise DPI for very small type (watch token/image size limits).
- **Few-shots**: Use crops from the **same printing or scribal hand**; include a line with **complex clusters + svaras** the model otherwise drops.
- **Prompts**: Name the notation system you want (e.g. **Udātta as acute**, **anudātta as underdots**, **svarita markers**), and forbid normalization unless you want diplomatic fidelity.
- **Provider choice**: Gemini Flash is fast for batch pages; Claude often excels on **ambiguous damaged glyphs** when cost/latency allow.

## Tests

```bash
cd backend
source .venv/bin/activate
pytest -q
```

## Project layout

- `backend/app` — FastAPI app, providers, PDF utils
- `backend/tests` — smoke tests
- `frontend` — Vite React UI
- `docker-compose.yml` — optional local stack
