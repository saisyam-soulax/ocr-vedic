# Incident Log

Downtime / failure log for the Vedic OCR pipeline: what broke, why, what
fixed it, and what's still open. Newest incidents first.

---

## INC-007 — "All pages successful" but ~40-60% of pages come back blank — hidden PDF layer rendered as blank by PyMuPDF

- **Date:** 2026-09-17 (job ran 2026-09-16)
- **Severity:** Medium — silent content loss with no error surfaced anywhere
- **Status:** ✅ Fixed, deployed

**Symptom**
"Shatpath Brahman III _part6.pdf" (94 pages) completed with `status:
complete, failed_count: 0` — every page reported successful — but roughly
half the pages (mostly a contiguous range from page ~38 onward, with a
handful of real-content pages interspersed) contained almost no text, just
an empty `<page id="001"></page>`-style stub.

**Investigation**
This job had exactly one clean submission (no resume, no concurrent
duplicate — ruled out INC-006 as the cause here). Per-page log lines showed
every page — blank or not — had essentially identical Gemini **input**
token counts (~5500), meaning a full, normal-sized image was sent for every
page; only the **output** token count collapsed to single digits for the
affected pages. That rules out an empty/failed request and points at
something about the image content itself.

Since the original PDF had already been deleted (per the standard
`UPLOAD_RETAIN=false` cleanup after a completed job), the actual images
were recovered from the still-present Vertex batch input
(`gs://svarupa-ocr-batch/batch-jobs/<job_id>/input.jsonl`, which Vertex
batch jobs leave in GCS well after completion) and decoded directly. The
extracted image for a "blank" page turned out to be a **pure white
1700×2200px canvas** — exactly 8.5×11in at 200 DPI, a generic Letter size,
nothing like the real scanned page's irregular ~1154×1828px dimensions
seen on a working page. Extending the check across 31 of the affected
pages showed **all 31 resolve to the exact same image (identical MD5
hash)** — the same single blank canvas, not 31 independently-blank scans.

That ruled out "many individually blank source pages happening to render
alike" — 31 independent renders coincidentally producing byte-identical
JPEGs is not plausible. It also ruled out an app-side indexing bug: the
page-number assignment loop
(`ocr_service.py`, `for page_num in range(1, n_pages + 1): page_slots.append(...)`)
is a plain, unconditional `range()` — every one of the 94 pages gets a
strictly unique, correctly incrementing page number with no possibility of
two slots requesting the same page or colliding. Since each of the 31
affected page numbers was rasterized independently (its own
`pdf_page_to_image()` call, its own fresh page load) and all still came
back byte-identical, the only remaining explanation is that **the uploaded
PDF itself has identical/blank content on those specific pages** — most
consistent with a padding/templating artifact from whatever process split
this "part6" chunk out of the larger master scan (e.g. the source ran out
of real pages for this chunk and the splitting tool filled the remainder
with a repeated blank Letter-sized page).

At this point the investigation had cleared this app's own code (page
numbering was provably unique per page) and initially pointed at the
uploaded file having genuinely blank/padded pages in that range. **The
user then confirmed they had checked the original PDF directly and it does
contain real text content on those pages** — ruling out "blank filler
pages in the file" and leaving one remaining explanation: a **hidden PDF
layer (Optional Content Group / OCG)**. Some PDFs place the actual scanned
page image on a layer marked hidden-by-default; many full-featured PDF
viewers show it anyway (depending on how they resolve default visibility),
but PyMuPDF's rendering respects the stored default and silently produces
a blank — but otherwise completely valid — page image. No exception, no
warning, nothing downstream (Gemini included) has any way to detect that
the "successful" page it received was empty by construction.

**Confirmed directly**, not just inferred: built a synthetic PDF with a
single hidden-by-default OCG layer containing the text "REAL PAGE CONTENT",
rendered it with PyMuPDF's default settings — the output was a blank white
image, byte-for-byte the same shape as this incident's affected pages.
Forcing that one layer visible (`doc.set_layer_ui_config(0, action=1)`)
and re-rendering produced the real text, visually confirmed. This
reproduces the exact failure mode outside of any specific customer file.

**Fix**
[`backend/app/utils/pdf.py`](../backend/app/utils/pdf.py): added
`_force_all_layers_visible(doc)`, which reads `doc.layer_ui_configs()` and
calls `set_layer_ui_config(number, action=1)` for every layer not already
on, immediately after opening the PDF. Called from both
`pdf_page_to_image()` (single-page path) and `pdf_bytes_to_page_images()`
(streaming path) before any page is rendered. For the overwhelming
majority of PDFs — which have no layers at all — `layer_ui_configs()`
returns an empty list and this is a complete no-op; it only does anything
for files that actually declare hidden OCGs.

Added [`tests/test_pdf_layers.py`](../backend/tests/test_pdf_layers.py)
with three cases: a synthetic PDF with hidden-layer text now renders
non-blank (both the single-page and multi-page rasterization paths), and a
normal layer-less PDF is confirmed unaffected (guards against a regression
where this "fix" itself breaks ordinary files). Full suite (48 tests)
passes. Rebuilt and redeployed; confirmed the fix function is present in
the live image.

**Not yet done:** re-run "Shatpath Brahman III _part6.pdf" (or the
specific page range) through the fixed pipeline to confirm the previously-
blank pages now come back with real transcribed text — this is strongly
expected given the reproduction above, but hasn't been directly confirmed
against the actual customer file (the originally-processed copy is gone
per this app's normal post-completion cleanup; a fresh upload of that same
source file would confirm it end-to-end).

---

## INC-006 — Concurrent resume/submit for the same job corrupts its results; last writer wins

- **Date:** 2026-09-16 (root cause dates to a resume race on 2026-09-15)
- **Severity:** High — user-visible data corruption (a genuinely completed
  job was shown as failed, with duplicated page content on disk)
- **Status:** ✅ Fixed, deployed

**Symptom**
A colleague's 150-page PDF ("Satapatha Brahmana part I_part3.pdf") showed as
not completed, a full day after it was submitted. Digging into the job's
history showed it had actually been submitted **three times** for the same
job ID within about an hour:
1. An initial submission went into a Vertex GCS batch job that sat polling
   for 2 hours with no visible progress in the UI — looking "stuck."
2. A **resume** was triggered (understandably, since it looked frozen),
   starting a second, independent batch run for the same job ID.
3. **18 seconds later**, another resume fired (likely a second click),
   starting a *third* independent run, which was then cancelled ~10 minutes
   in.

Run #2 finished successfully ~16 minutes later — all 150/150 pages — and the
job briefly showed complete (confirmed: it was even downloaded successfully
in that window). But **24 minutes after that**, run #1 (the original,
still silently polling in the background since hour zero) finally hit its
2-hour timeout and wrote its own `job_finished status=failed` — overwriting
the already-successful completion marker. From that point on the job showed
`failed`, even though the real OCR output was sitting right there on disk.

**Root cause**
`job_registry.py` tracks at most one `(queue, task, registered_at)` tuple per
job ID in a plain dict. `_start_ocr_task()` in `main.py` — called by both the
initial submit endpoint and `POST /api/ocr/{job_id}/resume` — **always**
created a brand-new `asyncio.Task` and called `register_job()` unconditionally,
with no check for whether a task was already running for that job ID.  The
existing resume-side guard (`if info.get('status') == 'complete': raise 409`)
only protects against resuming an *already-finished* job — it does nothing
for a job that's still actively running (exactly the "looks stuck" case that
prompted the resume in the first place).

Result: nothing stopped two or three independent tasks from running
concurrently against the same job directory. Each writes its own pages to
the same `results.jsonl` and, on completion, overwrites the same
`job_complete.json` — whichever task happens to finish **last** wins,
regardless of whether its result is the correct one. In this incident, the
stale, ultimately-timed-out original run happened to finish last and
clobbered a perfectly good result.

A secondary bug compounded this: the completion callback
(`_drop_job_when_done`) called `drop_job(job_id)` unconditionally whenever
*any* task for that job ID finished — including a stale task from an earlier,
already-superseded run — which could evict a different, currently-active
task's registry entry out from under it.

**Fix**
- [`backend/app/job_registry.py`](../backend/app/job_registry.py): added
  `is_job_active(job_id)`, returning `True` only when a registered task
  exists and hasn't finished yet.
- [`backend/app/main.py`](../backend/app/main.py) `_start_ocr_task()`: now
  checks `is_job_active(job_id)` **before** creating the new task, and raises
  `HTTPException(409, ...)` with a clear message if one is already running —
  for both the initial-submit and resume code paths, since both flow through
  this single function. A second submit/resume attempt while a job is in
  flight is now rejected outright instead of silently racing.
- Hardened `_drop_job_when_done` to only call `drop_job()` if the registry's
  *current* entry for that job ID is still the same task object that just
  finished — a stale/rejected task can no longer evict a newer active entry.

Verified: full test suite (45 tests, 3 new covering `is_job_active`) passes.
Live-tested the actual race by calling `_start_ocr_task()` twice back-to-back
for the same job ID with no `await` in between (so the first task has no
chance to run before the second call) — confirmed the second call is now
rejected with `409` and the intended message, whereas before this fix it
would have silently spawned a second concurrent task.

**Data repair performed for the affected job:** the two completed copies of
all 150 pages in `results.jsonl` were confirmed byte-identical (not
conflicting content — just duplicated), so the file was deduplicated back
to one copy per page, `ocr_output.txt` was regenerated, and
`job_complete.json` was corrected to reflect the true, successful completion
(150/150 pages, from the run that actually finished cleanly). Verified via
the live `/result`, `/download.txt`, and `/api/ocr/jobs` endpoints
afterward — all show the job as complete with clean, non-duplicated output.

**Not addressed (unchanged behavior):** a Vertex GCS batch job that's
genuinely progressing slowly still shows no visible progress in the UI until
it either completes or the 2-hour poll timeout is hit — that's what made
this job "look stuck" and prompted the resume in the first place. The fix
here prevents the *data corruption* that resulted from resuming a
still-running job; it does not (yet) give the UI a way to show that a batch
job is alive and progressing normally during a long quiet stretch. Worth a
future UX improvement — e.g., surfacing Vertex's own job state/progress in
the UI — so a slow-but-healthy job doesn't look indistinguishable from a
stuck one.

**Addendum (2026-09-17):** a second job hit by this exact same race was
found — `b045e768` ("Shatpath Brahman II _part1.pdf", 150 pages), resumed
twice 94 seconds apart on 2026-09-16 at 07:34/07:35, both resumes
completing successfully and both writing their full result sets, producing
the same "every page duplicated" pattern (300 lines / 150 unique in
`results.jsonl`, rendering as pages in the order 1,1,2,2,3,3,... once
sorted for display). Its last resume was ~1 hour *before* this incident's
fix was deployed (08:31 UTC on 2026-09-16), so it's an artifact of the
pre-fix window, not a recurrence. Confirmed the two duplicate blocks were
byte-identical (not conflicting), deduplicated `results.jsonl` back to one
copy per page, and rebuilt `ocr_output.txt` — `job_complete.json` for this
one was already correct (both duplicate runs had succeeded, so unlike the
job above, the last writer here happened to also be valid). Verified via
the live `/result` and `/download.txt` endpoints afterward.

---

## INC-005 — A single dropped connection while polling Vertex fails an otherwise-healthy batch job

- **Date:** 2026-08-31
- **Severity:** High — a completed, paid-for Gemini result was abandoned;
  the job showed "failed" while the work had actually succeeded
- **Status:** ✅ Fixed, deployed

**Symptom**
Job `359cbcdb` (a Gemini batch run of `Chandogya_Brahmanam_Sayana_5_sample_.pdf`)
had been polling normally for 38 minutes (`JOB_STATE_RUNNING`, steady
~2-minute poll cadence, no prior errors) and then abruptly finished with:
```
OCR job finished: job_id=359cbcdb-... status=failed elapsed=2320.97s pages_ok=0
```
Checking the live Vertex batch job immediately after showed it was **still
`JOB_STATE_RUNNING`** with no `end_time` — i.e. our app had declared the
job failed while Vertex was still actively working on it. ~3 minutes
later the Vertex job independently reached `JOB_STATE_SUCCEEDED`
(`end_time` 12:10:01 UTC) and its full output (`predictions.jsonl`,
4.66 MB) was sitting in the output GCS prefix — complete, paid-for, and
orphaned, because nothing in the app was still watching for it.

**Root cause**
The poll loops in `poll_gcs_batch()` and `poll_gemini_batch()` called
`client.batches.get(name=job_name)` directly with no error handling
around it. The actual traceback:
```
httpcore.ConnectError: [Errno 101] Network is unreachable
  File "app/providers/gemini_batch.py", line 417, in poll_gcs_batch
    job = await loop.run_in_executor(None, lambda: client.batches.get(name=job_name))
```
This is a **transient network blip from the container to
`aiplatform.googleapis.com`** during a routine status check — confirmed
recovered instantly on retest — not a Vertex-side failure. Because the
call had no retry, this single dropped connection propagated straight up
through `_run_gemini_batch()` and was caught by the job's generic
exception handler, which marked the whole OCR job `status=failed`. The
Vertex batch job itself was never touched — it was never cancelled, never
queried again, and kept running to completion with nobody left to
collect the result. Attempting to cancel it after the fact
(`client.batches.cancel(...)`) correctly failed with
`FAILED_PRECONDITION: ... is in state JOB_STATE_SUCCEEDED and cannot be
canceled` — it had already finished successfully by the time we checked.

This is a different failure mode from INC-003: INC-003 was Vertex
*itself* reporting a real terminal failure (queue-full) via the job's
`error` field, which is legitimate to fail fast on (after retrying the
transient cases). This incident was our own polling *client* losing
connectivity for a moment — the job's actual `state` was never even
successfully read that time, so there is no "job state" to inspect at
all; it's a plain network exception that must never be attributed to the
batch job.

**Fix**
[`backend/app/providers/gemini_batch.py`](../backend/app/providers/gemini_batch.py):
- Added `_is_transient_network_error()`, which recognizes `httpx.TransportError`,
  `ConnectionError`, `TimeoutError`, and `OSError` (and walks `__cause__`
  chains, since `google-genai` wraps httpx errors) — but explicitly does
  **not** match `google.genai.errors.ClientError`/`ServerError`, so a real
  API-level error (like INC-003's queue-full) still surfaces immediately
  without a pointless retry delay.
- Added `_poll_get_batch_job()`, which wraps `client.batches.get(...)`
  with up to 6 attempts and exponential backoff (5s → 10s → 20s → 40s →
  60s, capped), retrying only transient network errors. A persistent
  outage still eventually fails the job (correctly) after all attempts
  are exhausted; a momentary blip no longer does.
- Applied to **both** poll loops — `poll_gcs_batch()` (Vertex/GCS path)
  and `poll_gemini_batch()` (Developer API inline path) — since both had
  the identical unguarded call.

Verified with unit tests against fakes reproducing the exact exception
shape from the incident (`httpx.ConnectError`): confirms the retry fires
and recovers on transient errors, confirms a real `genai.errors.ClientError`
raises immediately with zero retries (so INC-003's fast-fail-on-real-failure
behavior is preserved), and confirms persistent network failure still
eventually raises after exhausting attempts. Full backend test suite (42
tests) passed after the change.

**Not yet addressed:** recovering already-orphaned output from a job that
fails this way — the fix prevents *future* occurrences, but a job that
already fell into this state before the fix has no automatic path back to
its now-abandoned Vertex output. `359cbcdb`'s output was manually
inspected and found intact in GCS; whether to build a recovery path is a
separate decision (see conversation for manual recovery follow-up).

---

## INC-004 — Oversized page geometry inflates batch payload 90x, upload never finishes

- **Date:** 2026-08-13
- **Severity:** High — batch jobs on affected PDFs could take 40+ min to
  upload and never reach Gemini within a practical wait
- **Status:** ✅ Fixed, deployed

**Symptom**
A 142-page PDF ("Maha Narayana Upanishad Bhashya...") submitted in Gemini
batch mode sat "uploading" for 38+ minutes with no progress to submission.
Meanwhile, 900-page documents submitted earlier the same day had uploaded
and completed quickly, making the failure look inexplicable by page count
alone.

**Root cause**
The PDF declared an oversized page box — **2810 × 4270 pt (39.0 × 59.3
inches)** — roughly 5x the linear dimensions of a normal book page, even
though the embedded scan was only ~2810×4270 px (12 MP). Rasterizing at the
configured 200 DPI multiplied this into a **7806 × 11862 px (92.6
megapixel)** image per page — pure upsampling with no real detail gained.
At JPEG quality 95, that's ~3.7 MB/page; across 142 pages plus the +33%
base64 overhead required for GCS batch input, the payload reached
**832.9 MB**. Combined with degraded host egress to `storage.googleapis.com`
(~0.3 MB/s, down from ~4.5 MB/s measured on an earlier job that day), the
upload simply couldn't finish in a reasonable time. This explains why the
"smaller" 142-page file was far heavier than the "larger" 900-page ones —
per-page size, not page count, drives payload size, and this file's
per-page size was ~10x normal due to the broken page box.

**Fix**
[`backend/app/utils/pdf.py`](../backend/app/utils/pdf.py) — added
`MAX_RENDER_EDGE_PX = 4000` and a `_capped_render_matrix()` helper used by
both `pdf_page_to_image()` and `pdf_bytes_to_page_images()`. Before
rendering, it checks whether the requested DPI would produce an image
whose longest edge exceeds 4000 px; if so, it scales the render matrix
down so the longest edge is capped at 4000 px (logging a warning with the
effective DPI). Normal-sized pages (e.g. 8.5×11in at 200 DPI → 2200 px)
never trigger the cap and are rendered exactly as before — this only
affects pathological page geometries.

Verified against a synthetic PDF combining a 39×59in page and a normal
8.5×11in page: the oversized page was capped from an 11800px long edge
down to 4000px (0.131 MB output) while the normal page rendered
unaffected (0.054 MB, no cap triggered). Full backend test suite (42
tests) passed after the change.

**Expected impact:** the same 142-page file would now produce roughly
90-110 MB of batch input instead of 833 MB — a small upload even on a
degraded link.

**Follow-up considered, not done:** the slow host egress to GCS
(~0.3 MB/s at the time of this incident, down from ~4.5 MB/s earlier that
day) is a separate infrastructure issue, likely contention from other
workloads on the same host. Worth a separate investigation if uploads
stay slow after this fix.

---

## INC-003 — Vertex "max queued batch jobs" quota silently fails jobs with no reason surfaced

- **Date:** 2026-08-12
- **Severity:** Medium — job fails with an opaque error; no cost incurred
- **Status:** ✅ Fixed, deployed

**Symptom**
A batch job uploaded its input to GCS successfully and was submitted to
Vertex, but the Vertex job transitioned to `JOB_STATE_FAILED` about 40
seconds after creation. The app only logged `"ended with state
JOB_STATE_FAILED"` — the real reason had to be pulled manually via the
Vertex API.

**Root cause**
Vertex AI enforces a per-project cap on the number of *queued* batch
prediction jobs. When the project's queue was momentarily full (likely
from other batch jobs sharing the GCP project), the newly submitted job
was rejected with `code=9: "Your project has reached the maximum number
of queued jobs."` before it ever started running (`start_time` was
`None` — confirmed via the Vertex API — so no inference occurred and no
cost was incurred). The poller in `poll_gcs_batch()` treated any terminal
failure state identically and discarded the actual `error` object from
the job response.

**Fix**
[`backend/app/providers/gemini_batch.py`](../backend/app/providers/gemini_batch.py):
1. On a terminal failure, the poller now reads `job.error` (code +
   message) and includes it in both the log line and the raised
   exception, so the real cause is visible without a manual API query.
2. Added `_is_queue_full_error()` to detect this specific transient
   condition (code 9/429, or "queued jobs" / "resource_exhausted" in the
   message).
3. Added a **bounded auto-retry**: when the queue-full error is detected,
   the poller waits with exponential backoff (30s → 60s → 120s → 240s,
   up to 5 attempts) and resubmits the batch job — reusing the
   already-uploaded GCS input via a new `_create_gcs_batch_job()` helper,
   so there's no re-upload. If still failing after all retries, it raises
   with the clear reason. This lives entirely in the poll loop, so
   healthy jobs pay zero extra latency.

Verified: confirmed via the live Vertex batch list that the failed job
was never actually running (`start_time=None`), and that the project's
batch queue was empty shortly after (108 succeeded / 2 failed / 0
active), meaning the condition was transient and self-clears.

---

## INC-002 — GCS batch input upload times out on slow/congested link, failing the whole job

- **Date:** 2026-08-12
- **Severity:** Medium — job fails before reaching Gemini; no cost incurred,
  but no automatic recovery
- **Status:** ✅ Fixed, deployed

**Symptom**
Three consecutive Gemini batch jobs failed within ~90 seconds of starting,
all with:
```
requests.exceptions.ConnectionError: ('Connection aborted.',
TimeoutError('The write operation timed out'))
  File "app/providers/gemini_batch.py", line 307, in _upload
    blob.upload_from_string(...)
```
Two of the three failing jobs were retries of the exact same file/page
count that had *succeeded* an hour earlier — the signature of a flaky
network link, not a bad input.

**Root cause**
`submit_gcs_batch()`'s `_upload()` called
`blob.upload_from_string(...)` with the client library's default timeout
(~60s) and no retry logic. Measured host egress to
`storage.googleapis.com` at the time was ~0.77 MB/s — slow enough that
larger batch inputs (100+ MB) routinely exceeded the default timeout on a
single write attempt, aborting the whole job.

Confirmed via direct verification that **no request had reached Gemini**
for any of the three failed jobs: the upload step runs strictly before
`client.batches.create()` in the code; none of the failed jobs logged
`"input uploaded, submitting job"`; and a live query of the Vertex batch
list showed zero jobs matching the failed jobs' display names. No cost
was incurred and resubmission was safe.

**Fix**
[`backend/app/providers/gemini_batch.py`](../backend/app/providers/gemini_batch.py)
`_upload()`: increased the per-attempt upload timeout to 600s and added
up to 4 attempts with exponential backoff (2s → 4s → 8s) around
`blob.upload_from_string(...)`. Also logs the payload size in MB before
uploading, so future slow uploads are visible in logs rather than only
showing up as a mysterious multi-minute wait.

---

## INC-001 — Non-ASCII filenames crash download endpoints (500 error)

- **Date:** 2026-08-07
- **Severity:** High — user-facing, blocked all downloads for affected jobs
- **Status:** ✅ Fixed, deployed

**Symptom**
Downloading `.txt` or `.docx` output failed with a 500 error for jobs
whose source filename contained Devanāgarī characters (e.g. "शाबर भाष्य
भाग 3 (2).pdf"), while jobs with ASCII filenames downloaded fine. Backend
logs showed:
```
File "app/main.py", line 921, in download_txt
UnicodeEncodeError: 'latin-1' codec can't encode characters in position 32-35
```

**Root cause**
Both `download_txt()` and `download_docx()` built the `Content-Disposition`
response header by interpolating the source filename directly:
`f'attachment; filename="vedic-ocr-{slug}.txt"'`. HTTP header values must
be latin-1 encodable; a non-ASCII (e.g. Devanāgarī) filename made the
header impossible to encode, and Starlette raised before the response
could be sent — for *every* download of that job, not just once.

**Fix**
[`backend/app/main.py`](../backend/app/main.py): added
`_content_disposition()`, which builds an RFC 5987–compliant header: an
ASCII-safe `filename=` fallback (non-ASCII chars replaced with `?`) plus a
`filename*=UTF-8''<percent-encoded>` parameter carrying the real name.
Modern browsers use the `filename*` form, so the saved file still gets
the correct Devanāgarī name; older clients fall back gracefully to the
ASCII name. Applied to both `download_txt()` and `download_docx()`.

Verified against the actual failing job: both endpoints returned `200`
after the fix, with a latin-1-safe header confirmed by direct encoding
test.

---

## Recurring operational notes (not incidents, but worth knowing)

- **The backend does not auto-resume incomplete jobs on startup.**
  `lifespan()` in `main.py` only logs and yields; resuming a job requires
  an explicit `/resume` call. This means restarting the backend is a safe
  way to kill a stuck/wedged job — it won't come back on its own, it just
  becomes a resumable entry in Recent Jobs.
- **Rapid cancel → resume → cancel on the same job can wedge the
  backend.** Observed on 2026-08-13: a 142-page job was cancelled 3x and
  resumed 2x within ~2 minutes, after which the backend's event loop
  became starved (simple `/health` calls took 10-19s instead of ~1ms) and
  CPU pegged at ~104% for 6+ minutes with no progress. Suspected cause:
  PDF rasterization runs synchronously and can overlap across
  cancel/resume cycles, starving the async event loop. **Not yet fixed —
  logged here as a known follow-up.** Restarting the backend is the
  current workaround.
- **A job "failing" or "queued" at the GCS-upload or Vertex-submission
  stage never incurs Gemini cost.** Cost only occurs once a batch job
  actually starts running on Vertex (`start_time` is set) or an inline
  request completes. This has been directly verified via the Vertex
  batch job list multiple times during these incidents — always check
  `start_time` and the live batch list before assuming a failed job cost
  anything.
