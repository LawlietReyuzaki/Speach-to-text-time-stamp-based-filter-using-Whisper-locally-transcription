"""FastAPI app. The API validates, stores, creates a job row, enqueues, and returns."""
from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Callable, Iterator, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

from app import auth
from app import db as dbm
from app import formats
from app.config import ALLOWED_EXTENSIONS, UPLOAD_CHUNK_BYTES, Settings, load_settings
from app.db import Database, Job
from app.languages import LANGUAGES
from app.queue import InMemoryQueue, QueueFull
from app.storage import LocalStorage, Storage
from app.transcribe import Transcriber, WhisperTranscriber
from app.worker import WorkerPool

log = logging.getLogger(__name__)
# threading.Lock is a factory function before 3.13, so the annotation must stay a string.
TranscriberFactory = Callable[[Optional["threading.Lock"]], Transcriber]


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(status_code=status, detail={"code": code, "message": message}, headers=headers)


class TokenBucket:
    """Per-key rate limit: `per_minute` tokens, refilled continuously."""

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            return True


class TooLarge(Exception):
    pass


def _read_pieces(fh: BinaryIO, max_bytes: int, digest: "hashlib._Hash") -> Iterator[bytes]:
    """Yield 1 MB pieces, hashing as we go, and stop early when the limit is passed."""
    total = 0
    fh.seek(0)
    while piece := fh.read(UPLOAD_CHUNK_BYTES):
        total += len(piece)
        if total > max_bytes:
            raise TooLarge
        digest.update(piece)
        yield piece


def _save_upload(storage: Storage, key: str, fh: BinaryIO, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = storage.write(key, _read_pieces(fh, max_bytes, digest))
    return digest.hexdigest(), size


def create_app(settings: Settings | None = None, make_transcriber: TranscriberFactory | None = None) -> FastAPI:
    settings = settings or load_settings()

    def default_factory(lock: threading.Lock | None) -> Transcriber:
        return WhisperTranscriber(
            settings.model_size, settings.compute_type, settings.device, settings.chunk_timeout_sec, lock
        )

    factory = make_transcriber or default_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        db = Database(settings.db_path)
        storage = LocalStorage(settings.data_dir)
        queue = InMemoryQueue(settings.queue_size)
        pool = WorkerPool(settings, db, queue, storage, factory)
        app.state.db, app.state.storage, app.state.queue, app.state.pool = db, storage, queue, pool
        app.state.limiter = TokenBucket(settings.rate_limit_per_min)
        pool.recover()
        pool.start()
        yield
        # SIGTERM: uvicorn stops accepting requests, then we drain the workers.
        pool.stop()
        db.close()

    app = FastAPI(title="transcription-service", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": str(exc.detail)}
        return JSONResponse({"error": detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": {"code": "invalid_input", "message": str(exc)}}, status_code=400)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"error": {"code": "internal", "message": "internal server error"}}, status_code=500)

    def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
            raise ApiError(401, "unauthorized", "missing or invalid X-API-Key")
        return x_api_key

    def require_auth(
        authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)
    ) -> str:
        """Bearer JWT (from /v1/auth/token) or the raw API key. Returns the caller identity."""
        if authorization and authorization.lower().startswith("bearer "):
            try:
                return str(auth.verify_token(settings.jwt_secret, authorization[7:].strip())["sub"])
            except auth.InvalidToken as exc:
                raise ApiError(401, "unauthorized", str(exc)) from exc
        return require_api_key(x_api_key)

    def get_job_or_404(job_id: str, request: Request) -> Job:
        job = request.app.state.db.get_job(job_id)
        if job is None:
            raise ApiError(404, "not_found", f"job {job_id} not found")
        return job

    def enqueue_or_429(request: Request, job_id: str) -> None:
        try:
            request.app.state.queue.publish(job_id)
        except QueueFull as exc:
            raise ApiError(429, "queue_full", str(exc), headers={"Retry-After": "30"}) from exc

    # ---- routes -----------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Small browser UI for manual testing; the API key is entered on the page."""
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.post("/v1/auth/token")
    def issue_token(api_key: str = Depends(require_api_key)) -> dict[str, Any]:
        token = auth.create_token(settings.jwt_secret, subject="api-key", ttl_sec=settings.jwt_ttl_sec)
        return {"access_token": token, "token_type": "bearer", "expires_in": settings.jwt_ttl_sec}

    @app.post("/v1/transcriptions")
    async def create_transcription(
        request: Request,
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
        caller: str = Depends(require_auth),
    ) -> JSONResponse:
        if not request.app.state.limiter.allow(caller):
            raise ApiError(429, "rate_limited", "too many uploads", headers={"Retry-After": "60"})
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ApiError(400, "invalid_input", f"unsupported file type {ext!r}")
        if language and language not in LANGUAGES:
            raise ApiError(400, "invalid_input", f"unknown language code {language!r}, see /v1/languages")
        declared = int(request.headers.get("content-length", 0))
        if declared > settings.max_upload_bytes:
            raise ApiError(413, "too_large", f"upload exceeds {settings.max_upload_bytes} bytes")

        db: Database = request.app.state.db
        storage: Storage = request.app.state.storage
        job_id = uuid.uuid4().hex
        key = f"{job_id}/original{ext}"
        try:
            sha256, size = await run_in_threadpool(_save_upload, storage, key, file.file, settings.max_upload_bytes)
        except TooLarge:
            storage.delete_prefix(job_id)
            raise ApiError(413, "too_large", f"upload exceeds {settings.max_upload_bytes} bytes")
        if size == 0:
            storage.delete_prefix(job_id)
            raise ApiError(400, "invalid_input", "uploaded file is empty")

        filename = Path(file.filename or key).name
        done = db.find_by_hash(sha256, settings.model_size, [dbm.DONE])
        if done is not None:
            # Same bytes, same model: point at the existing transcript, skip the work.
            storage.delete_prefix(job_id)
            db.create_job(job_id, done.storage_key, sha256, filename, settings.model_size,
                          done.language, status=dbm.DONE, result_of=done.id)
            return JSONResponse({"job_id": job_id, "status": dbm.DONE}, status_code=200)
        active = db.find_by_hash(sha256, settings.model_size, [dbm.QUEUED, dbm.PROCESSING])
        if active is not None:
            storage.delete_prefix(job_id)
            return JSONResponse({"job_id": active.id, "status": active.status}, status_code=200)

        db.create_job(job_id, key, sha256, filename, settings.model_size, language)
        try:
            enqueue_or_429(request, job_id)
        except ApiError:
            db.delete_job(job_id)
            storage.delete_prefix(job_id)
            raise
        return JSONResponse({"job_id": job_id, "status": dbm.QUEUED}, status_code=202)

    @app.get("/v1/transcriptions/{job_id}")
    def get_transcription(
        job_id: str, request: Request, format: str | None = None, caller: str = Depends(require_auth)
    ) -> Response:
        job = get_job_or_404(job_id, request)
        db: Database = request.app.state.db
        source = db.get_job(job.result_of) if job.result_of else job
        if source is None:
            raise ApiError(404, "not_found", "referenced transcript was deleted")
        body: dict[str, Any] = {
            "job_id": job.id,
            "status": job.status,
            "attempts": job.attempts,
            "error": job.error,
            "result_of": job.result_of,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        if job.status != dbm.DONE:
            if format:
                raise ApiError(409, "not_ready", f"job is {job.status}, transcript not available")
            return JSONResponse(body)
        segments = db.segments(source.id)
        if format == "srt":
            return PlainTextResponse(formats.to_srt(segments))
        if format == "vtt":
            return PlainTextResponse(formats.to_vtt(segments), media_type="text/vtt")
        if format == "txt":
            return PlainTextResponse(formats.to_txt(segments))
        if format:
            raise ApiError(400, "invalid_input", "format must be srt, vtt or txt")
        body.update(formats.to_json(job.filename, source.language, source.duration or 0.0, segments))
        return JSONResponse(body)

    @app.delete("/v1/transcriptions/{job_id}")
    def delete_transcription(job_id: str, request: Request, caller: str = Depends(require_auth)) -> Response:
        job = get_job_or_404(job_id, request)
        db: Database = request.app.state.db
        if job.status == dbm.PROCESSING:
            # The worker sees 'cancelled' between chunks and removes the files itself.
            db.cancel_job(job_id)
            return JSONResponse({"job_id": job_id, "status": dbm.CANCELLED}, status_code=202)
        db.delete_job(job_id)
        request.app.state.storage.delete_prefix(job_id)
        return Response(status_code=204)

    @app.get("/v1/languages")
    def list_languages() -> dict[str, Any]:
        return {"auto": "detect automatically", "languages": LANGUAGES}

    @app.get("/v1/transcriptions/{job_id}/audio")
    def download_audio(job_id: str, request: Request, caller: str = Depends(require_auth)) -> FileResponse:
        job = get_job_or_404(job_id, request)
        path = request.app.state.storage.local_path(job.storage_key)
        if not path.is_file():
            raise ApiError(404, "not_found", "audio file no longer stored")
        return FileResponse(path, filename=job.filename)

    @app.post("/v1/transcriptions/{job_id}/retry")
    def retry_transcription(job_id: str, request: Request, caller: str = Depends(require_auth)) -> JSONResponse:
        get_job_or_404(job_id, request)
        if not request.app.state.db.retry_failed_job(job_id):
            raise ApiError(409, "not_failed", "only failed jobs can be retried")
        enqueue_or_429(request, job_id)
        return JSONResponse({"job_id": job_id, "status": dbm.QUEUED}, status_code=202)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        pool: WorkerPool = request.app.state.pool
        return {
            "status": "ok",
            "queue_size": request.app.state.queue.size(),
            "active_workers": pool.active,
            "workers": settings.workers,
            "model": {"name": f"whisper-{settings.model_size}", "device": settings.device,
                      "compute_type": settings.compute_type},
        }

    return app


app = create_app()
