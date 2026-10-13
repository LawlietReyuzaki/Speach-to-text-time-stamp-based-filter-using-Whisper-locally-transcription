"""The transcription pipeline (shared with cli.py) and the worker pool that runs it."""
from __future__ import annotations

import logging
import random
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app import audio
from app import db as dbm
from app.config import Settings
from app.db import Database, Job
from app.errors import JobInterrupted, PermanentError, TransientError
from app.queue import Queue, QueueFull
from app.storage import Storage
from app.transcribe import Segment, Transcriber

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    language: str | None
    duration: float
    segments: list[Segment]  # absolute timestamps, only for chunks processed in this run


def run_pipeline(
    src: Path,
    work_dir: Path,
    transcriber: Transcriber,
    settings: Settings,
    language: str | None = None,
    skip_chunks: frozenset[int] = frozenset(),
    before_chunk: Callable[[int], None] = lambda i: None,
    on_chunk_done: Callable[[int, str, list[Segment]], None] = lambda i, lang, segs: None,
) -> PipelineResult:
    """Normalize, chunk, transcribe. Hooks let the worker checkpoint and cancel."""
    wav = audio.normalize(src, work_dir / "normalized.wav", settings.ffmpeg_bin, settings.ffmpeg_timeout_sec)
    duration = audio.wav_duration(wav)
    chunks = audio.split(wav, work_dir / "chunks", settings.chunk_sec, settings.ffmpeg_bin, settings.ffmpeg_timeout_sec)

    offset = 0.0
    segments: list[Segment] = []
    for index, chunk in enumerate(chunks):
        before_chunk(index)
        if index not in skip_chunks:
            result = transcriber.transcribe(chunk, language)
            # Language is detected on the first chunk and pinned for the rest.
            language = language or result.language
            shifted = [Segment(s.start + offset, s.end + offset, s.text) for s in result.segments]
            segments.extend(shifted)
            on_chunk_done(index, language, shifted)
        # Real chunk length, not the nominal CHUNK_SEC: the last chunk is shorter.
        offset += audio.wav_duration(chunk)
    return PipelineResult(language=language, duration=duration, segments=segments)


class WorkerPool:
    """Fixed number of worker threads plus one sweeper. Never one thread per request."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        queue: Queue,
        storage: Storage,
        make_transcriber: Callable[[threading.Lock | None], Transcriber],
    ) -> None:
        self.settings = settings
        self.db = db
        self.queue = queue
        self.storage = storage
        self._make_transcriber = make_transcriber
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._active = 0
        self._active_lock = threading.Lock()
        self._shared: Transcriber | None = None

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self.settings.share_model and self.settings.workers:
            self._shared = self._make_transcriber(threading.Lock())
        for n in range(self.settings.workers):
            t = threading.Thread(target=self._worker_loop, name=f"worker-{n}", daemon=True)
            t.start()
            self._threads.append(t)
        sweeper = threading.Thread(target=self._sweeper_loop, name="sweeper", daemon=True)
        sweeper.start()
        self._threads.append(sweeper)

    def stop(self) -> None:
        """Let the current chunk finish; workers requeue their job between chunks."""
        self._stop.set()
        for t in self._threads:
            t.join()

    def recover(self) -> None:
        """On startup the DB is the truth: requeue stale jobs and re-trigger queued ones."""
        for job_id in self.db.requeue_stale(self.settings.stale_sec, self.settings.max_attempts):
            log.warning("job %s was processing when the service died, requeued", job_id)
        for job_id in self.db.job_ids_with_status(dbm.QUEUED):
            self._publish_quietly(job_id)

    @property
    def active(self) -> int:
        with self._active_lock:
            return self._active

    # ---- loops ------------------------------------------------------------

    def _worker_loop(self) -> None:
        worker_id = f"{threading.current_thread().name}-{uuid.uuid4().hex[:8]}"
        transcriber = self._shared or self._make_transcriber(None)
        while not self._stop.is_set():
            job_id = self.queue.consume(timeout=0.5)
            if job_id is None:
                continue
            with self._active_lock:
                self._active += 1
            try:
                self.process(job_id, worker_id, transcriber)
            finally:
                with self._active_lock:
                    self._active -= 1
        self.db.close()

    def _sweeper_loop(self) -> None:
        while not self._stop.wait(self.settings.sweep_interval_sec):
            try:
                self.sweep()
            except Exception:  # a sweeper crash must not take recovery down
                log.exception("sweeper failed")
        self.db.close()

    def sweep(self) -> None:
        """Requeue jobs whose worker died, and re-trigger queued jobs nobody picked up."""
        for job_id in self.db.requeue_stale(self.settings.stale_sec, self.settings.max_attempts):
            log.warning("job %s had a stale heartbeat, requeued", job_id)
            self._publish_quietly(job_id)
        # Duplicate triggers are harmless because claim_job is atomic.
        cutoff = time.time() - self.settings.stale_sec
        for job_id in self.db.job_ids_with_status(dbm.QUEUED, updated_before=cutoff):
            self._publish_quietly(job_id)

    def _publish_quietly(self, job_id: str) -> None:
        try:
            self.queue.publish(job_id)
        except QueueFull:
            log.warning("queue full, job %s stays queued in the DB until the next sweep", job_id)

    # ---- one job ----------------------------------------------------------

    def process(self, job_id: str, worker_id: str, transcriber: Transcriber) -> None:
        if not self.db.claim_job(job_id, worker_id):
            log.info("job %s not claimable (duplicate message or cancelled), skipping", job_id)
            self.queue.ack(job_id)
            return
        job = self.db.get_job(job_id)
        assert job is not None
        work_dir = self.settings.data_dir / job_id / "work"
        try:
            result = self._run_job(job, work_dir, transcriber)
            if not self.db.finish_job(job_id, result.language or "unknown", result.duration):
                raise JobInterrupted("cancelled")  # DELETE arrived during the last chunk
            log.info("job %s done in %d attempt(s)", job_id, job.attempts)
            self.queue.ack(job_id)
        except JobInterrupted as exc:
            self._on_interrupted(job_id, str(exc))
        except PermanentError as exc:
            log.warning("job %s failed permanently: %s", job_id, exc)
            self.db.fail_job(job_id, str(exc))
            self.queue.ack(job_id)
        except Exception as exc:  # TransientError and anything unexpected
            self._on_transient(job, exc)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_job(self, job: Job, work_dir: Path, transcriber: Transcriber) -> PipelineResult:
        def before_chunk(index: int) -> None:
            if self._stop.is_set():
                raise JobInterrupted("shutdown")
            if self.db.get_status(job.id) != dbm.PROCESSING:
                raise JobInterrupted("cancelled")

        def on_chunk_done(index: int, language: str, segments: list[Segment]) -> None:
            self.db.upsert_segments(job.id, index, segments)
            self.db.set_language(job.id, language)
            self.db.heartbeat(job.id)

        return run_pipeline(
            self.storage.local_path(job.storage_key),
            work_dir,
            transcriber,
            self.settings,
            language=job.language,
            skip_chunks=frozenset(self.db.done_chunks(job.id)),
            before_chunk=before_chunk,
            on_chunk_done=on_chunk_done,
        )

    def _on_interrupted(self, job_id: str, reason: str) -> None:
        if reason == "shutdown":
            # Back to queued without touching attempts; startup recovery re-triggers it.
            self.db.requeue_job(job_id)
            log.info("job %s requeued for shutdown", job_id)
        else:
            # DELETE marked it cancelled while we were running; finish the cleanup.
            self.storage.delete_prefix(job_id)
            self.db.delete_job(job_id)
            log.info("job %s cancelled, files removed", job_id)
        self.queue.ack(job_id)

    def _on_transient(self, job: Job, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, TransientError):
            log.exception("job %s crashed unexpectedly", job.id)
        if job.attempts >= self.settings.max_attempts:
            log.warning("job %s failed after %d attempts: %s", job.id, job.attempts, error)
            self.db.fail_job(job.id, error)
            self.queue.ack(job.id)
            return
        delay = self.settings.retry_base_sec * (2 ** (job.attempts - 1)) * random.uniform(1.0, 1.5)
        log.warning("job %s attempt %d failed (%s), retry in %.1fs", job.id, job.attempts, error, delay)
        self.db.requeue_job(job.id, error=error)
        # Delayed nack keeps the worker free during the backoff.
        timer = threading.Timer(delay, self.queue.nack, args=(job.id,))
        timer.daemon = True
        timer.start()
