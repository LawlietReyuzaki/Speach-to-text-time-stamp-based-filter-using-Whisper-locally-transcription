# transcription-service

Small audio transcription service: FastAPI in front, a fixed pool of worker threads behind,
faster-whisper for speech to text, ffmpeg for decoding, SQLite for state, local disk for files.

```
app/main.py        FastAPI app, routes, startup/shutdown
app/config.py      env based settings
app/storage.py     Storage interface + LocalStorage
app/db.py          sqlite helpers, schema, atomic state transitions
app/queue.py       Queue interface + InMemoryQueue (PubSubQueue stub)
app/worker.py      run_pipeline (shared with the CLI) + WorkerPool (workers, sweeper)
app/audio.py       validate, normalize to wav, split into chunks
app/transcribe.py  faster-whisper wrapper, returns segments with timestamps
app/errors.py      PermanentError / TransientError / JobInterrupted
app/formats.py     json / srt / vtt / txt output
app/auth.py        HS256 JWT mint/verify (stdlib only)
app/languages.py   whisper language codes -> names, served at /v1/languages
app/static/        index.html, a small browser UI served at /
cli.py             python cli.py file.mp3 -> prints json
tests/             pytest, model is faked, audio is a generated sine tone
samples/sample.mp3 10 s spoken sample for the end-to-end check
samples/sentences.mp3  7 short sentences spoken fast, shows per-sentence timing
```

## Run

Requires Python 3.11+ and `ffmpeg` on PATH (or set `FFMPEG_BIN`).

The browser UI at `/` can record from the microphone, lets you pick a language (or auto-detect),
shows file details before you confirm, and offers the stored audio and transcript for download.

```bash
pip install -r requirements.txt

# standalone
python cli.py samples/sample.mp3

# api (then open http://localhost:8000 for the browser UI, or /docs for swagger)
API_KEY=dev-key uvicorn app.main:app --port 8000

# get a JWT (valid JWT_TTL_SEC, default 1 h); every /v1 route also accepts the raw X-API-Key
TOKEN=$(curl -s -X POST localhost:8000/v1/auth/token -H "X-API-Key: dev-key" | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
# upload (202 {job_id, status})
curl -s -X POST localhost:8000/v1/transcriptions -H "Authorization: Bearer $TOKEN" -F file=@samples/sample.mp3
# poll
curl -s localhost:8000/v1/transcriptions/<job_id> -H "X-API-Key: dev-key"
# subtitles
curl -s "localhost:8000/v1/transcriptions/<job_id>?format=srt" -H "X-API-Key: dev-key"
# retry a failed job / delete / health
curl -s -X POST localhost:8000/v1/transcriptions/<job_id>/retry -H "X-API-Key: dev-key"
curl -s -X DELETE localhost:8000/v1/transcriptions/<job_id> -H "X-API-Key: dev-key"
curl -s localhost:8000/health

python -m pytest -q
```

Settings (env vars, defaults in `app/config.py`): `API_KEY`, `JWT_SECRET`, `JWT_TTL_SEC=3600`, `DATA_DIR`, `MODEL_SIZE=small`,
`COMPUTE_TYPE=int8`, `DEVICE=cpu`, `WORKERS=2`, `SHARE_MODEL=false`, `QUEUE_SIZE=100`,
`CHUNK_SEC=600`, `MAX_UPLOAD_MB=200`, `MAX_ATTEMPTS=3`, `RETRY_BASE_SEC=2`, `STALE_SEC=300`,
`SWEEP_SEC=30`, `FFMPEG_BIN`, `FFMPEG_TIMEOUT_SEC=300`, `CHUNK_TIMEOUT_SEC=900`, `RATE_LIMIT_PER_MIN=10`.

GPU: `DEVICE=cuda COMPUTE_TYPE=float16`. On Windows also
`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`; the app adds their DLL folders to PATH
itself. With `SHARE_MODEL=true` all workers share one model instance behind a lock, which keeps
VRAM use to a single copy.

Errors always look like `{"error": {"code": "...", "message": "..."}}`:
400 bad input, 401 bad key, 404 unknown job, 409 wrong state, 413 too large, 429 queue full or
rate limited (with `Retry-After`).

## Design notes

**How do you handle different audio formats?**
The API and CLI only check the extension (wav, mp3, m4a, flac, ogg, webm), size, and emptiness up
front. The real decoding is delegated to ffmpeg, which converts anything it understands into
16 kHz mono 16-bit PCM wav, the format whisper wants. The extension is never trusted for
content: a `.mp3` full of garbage makes ffmpeg exit non-zero, which becomes a `PermanentError`
and a `failed` job with the ffmpeg message stored. The normalized wav is also where duration
comes from, read straight from the wav header.

**How are segments formed?**
Whisper runs with word timestamps on, and the words are regrouped into sentences on
sentence-ending punctuation (`. ! ?` plus Urdu `۔ ؟`, Devanagari `।`, CJK `。！？`). Each segment is
one sentence, timed from its first word to its last word, instead of whatever phrase whisper
happened to decode in one go.

**How do you deal with long audio files?**
If the normalized audio is longer than `CHUNK_SEC` (10 minutes), ffmpeg's segment muxer cuts
it into chunk files. Each chunk is transcribed on its own and its timestamps are shifted by the
running offset, which is computed from the real length of each chunk file, so the last, shorter
chunk does not break alignment. The language is detected on the first chunk and pinned for the
rest. After every chunk the segments are upserted into SQLite, so a retry skips chunks that are
already stored and continues from the first unfinished one. The model is loaded once per worker,
never per chunk.

**How would you handle concurrent uploads?**
The API never transcribes. It streams the upload to disk in 1 MB pieces (hashing as it goes),
inserts a `queued` job row, publishes the job id to a bounded queue and returns 202. Every job
has its own folder (`data/{job_id}/`), so uploads share nothing. A fixed pool of `WORKERS`
threads consumes the queue; each claims a job with one atomic `UPDATE ... WHERE status='queued'`,
so two workers (or a duplicate queue message) can never process the same job. If the queue is
full the upload is deleted again and the client gets 429 with `Retry-After`; a per-key token
bucket also limits uploads per minute.

**How would you store audio and transcripts?**
Audio goes through the `Storage` interface; `LocalStorage` writes under `DATA_DIR`. Transcripts
live in SQLite (WAL mode, `busy_timeout`, one connection per thread, short `BEGIN IMMEDIATE`
transactions, never a transaction open while transcribing). Segments are keyed by
`(job_id, chunk_index, seg_index)` and written with an upsert, so re-running a chunk or finishing
a job twice yields identical rows. Uploads are hashed with SHA-256; a second upload of the same
bytes with the same model gets a new job row that points to the existing transcript
(`result_of`), and an upload that is still in flight returns the existing job id.

**How do you retry or recover failed transcriptions?**
Errors are split in two classes. `PermanentError` (corrupt file, no audio, unsupported) fails
the job immediately. `TransientError` and any unexpected exception are retried with exponential
backoff and jitter (2 s, 4 s, 8 s) up to `MAX_ATTEMPTS`; after that the job is `failed` with the
last error stored and can be inspected or requeued through `POST .../retry`. Workers write a
heartbeat after every chunk; a sweeper thread runs every 30 s and requeues `processing` jobs
whose heartbeat is older than 5 minutes (a crashed worker), which then resume from their
checkpoint. On SIGTERM workers finish their current chunk and hand the job back to `queued`; on
startup the service runs the sweeper once and re-publishes every `queued` job, because the
database is the source of truth and the queue is only a trigger.

**How would you expose this as an API?**
`POST /v1/transcriptions` (multipart, optional `language`) returns 202 with a job id.
`GET /v1/transcriptions/{id}` returns status, attempts and error, plus language, duration,
segments and full text once done; `?format=srt|vtt|txt` returns the same transcript as
subtitles or plain text. `DELETE` removes files and rows, or marks a running job `cancelled` so
the worker stops at the next chunk boundary and cleans up. `GET /health` reports queue size and
active workers. Auth: `POST /v1/auth/token` exchanges the `X-API-Key` for a short-lived HS256
JWT (`JWT_SECRET`, `JWT_TTL_SEC`) sent as `Authorization: Bearer`; the raw key is accepted too
for scripts. Every error has the same JSON shape.

## Going to production

- **Queue:** replace `InMemoryQueue` with `PubSubQueue` (the stub in `app/queue.py` documents
  the mapping): `publish` writes the job id to a topic, `consume` pulls from a pull
  subscription, `ack`/`nack` map directly, and the subscription's ack deadline is the lease,
  extended between chunks where we already heartbeat. A dead-letter topic receives messages
  that exceed the maximum delivery attempts, while the job row keeps `failed` for inspection.
  Because the worker claims with an atomic update, Pub/Sub's at-least-once delivery needs no
  other change outside `queue.py`.
- **Storage:** replace `LocalStorage` with a GCS implementation. Clients upload straight to a
  presigned (signed) URL, then call the API with the object key, so audio bytes never pass
  through the API process; the worker downloads to its scratch dir in `local_path`.
- **Database:** move to Postgres. The claim becomes
  `SELECT ... FROM jobs WHERE status='queued' FOR UPDATE SKIP LOCKED`, which lets many worker
  processes pull work safely; everything else (upserts, heartbeats, stale sweep) is the same SQL.
- **Workers:** run them as separate processes or containers on GPU machines
  (`DEVICE=cuda`, `COMPUTE_TYPE=float16`), scaled on queue depth. The API tier stays small and
  stateless.
- **Chunking:** cut on silence (ffmpeg `silencedetect` or the VAD timestamps) instead of fixed
  10 minute blocks, so words are never split across chunk boundaries.
