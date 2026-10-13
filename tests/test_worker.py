"""Worker behaviour driven directly through db + queue + pool (no HTTP)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest

from app import db as dbm
from app.config import Settings
from app.db import Database
from app.errors import PermanentError, TransientError
from app.queue import InMemoryQueue
from app.storage import LocalStorage
from app.worker import WorkerPool
from tests.conftest import FakeTranscriber, make_settings


class Harness:
    def __init__(self, settings: Settings, fake: FakeTranscriber) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.storage = LocalStorage(settings.data_dir)
        self.queue = InMemoryQueue(settings.queue_size)
        self.pool = WorkerPool(settings, self.db, self.queue, self.storage, lambda lock: fake)

    def add_job(self, audio: Path, job_id: str = "job1") -> str:
        key = f"{job_id}/original{audio.suffix}"
        self.storage.write(key, [audio.read_bytes()])
        self.db.create_job(job_id, key, f"sha-{job_id}", audio.name, self.settings.model_size, None)
        return job_id

    def wait(self, job_id: str, statuses: set[str], timeout: float = 10.0) -> dbm.Job:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.db.get_job(job_id)
            if job is not None and job.status in statuses:
                return job
            time.sleep(0.02)
        raise AssertionError(f"{job_id} never reached {statuses}")


@pytest.fixture
def harness(tmp_path: Path, fake: FakeTranscriber) -> Iterator[Harness]:
    h = Harness(make_settings(tmp_path), fake)
    h.pool.start()
    yield h
    h.pool.stop()


def test_duplicate_queue_message_is_processed_once(harness: Harness, fake: FakeTranscriber, tone_wav: Path) -> None:
    job_id = harness.add_job(tone_wav)
    harness.queue.publish(job_id)
    harness.queue.publish(job_id)  # at-least-once delivery

    job = harness.wait(job_id, {dbm.DONE})
    time.sleep(0.2)  # give the duplicate a chance to be (not) processed
    assert job.attempts == 1
    assert fake.calls == ["normalized.wav"]
    assert harness.queue.size() == 0


def test_permanent_error_fails_without_retry(harness: Harness, fake: FakeTranscriber, tone_wav: Path) -> None:
    fake.fail = lambda n: PermanentError("no speech decodable")
    job_id = harness.add_job(tone_wav)
    harness.queue.publish(job_id)

    job = harness.wait(job_id, {dbm.FAILED})
    assert job.attempts == 1
    assert "no speech decodable" in (job.error or "")
    assert len(fake.calls) == 1


def test_transient_error_retries_up_to_max_then_fails(harness: Harness, fake: FakeTranscriber, tone_wav: Path) -> None:
    fake.fail = lambda n: TransientError("gpu hiccup")
    job_id = harness.add_job(tone_wav)
    harness.queue.publish(job_id)

    job = harness.wait(job_id, {dbm.FAILED})
    assert job.attempts == harness.settings.max_attempts == 3
    assert len(fake.calls) == 3
    assert "gpu hiccup" in (job.error or "")


def test_unexpected_exception_is_treated_as_transient(harness: Harness, fake: FakeTranscriber, tone_wav: Path) -> None:
    fake.fail = lambda n: RuntimeError("boom") if n == 1 else None
    job_id = harness.add_job(tone_wav)
    harness.queue.publish(job_id)

    job = harness.wait(job_id, {dbm.DONE})
    assert job.attempts == 2


def test_resume_from_checkpoint_after_crash_at_chunk_2(harness: Harness, fake: FakeTranscriber, long_wav: Path) -> None:
    # long_wav has 3 chunks. Crash on the second call (chunk index 1); chunk 0 is checkpointed.
    fake.fail = lambda n: TransientError("simulated crash") if n == 2 else None
    job_id = harness.add_job(long_wav)
    harness.queue.publish(job_id)

    job = harness.wait(job_id, {dbm.DONE})
    assert job.attempts == 2
    assert fake.calls == ["chunk_0000.wav", "chunk_0001.wav", "chunk_0001.wav", "chunk_0002.wav"]
    segments = harness.db.segments(job_id)
    assert [s.text for s in segments] == ["text of chunk_0000", "text of chunk_0001", "text of chunk_0002"]
    assert segments[-1].end == pytest.approx(2.5, abs=0.1)


def test_shutdown_requeues_job_between_chunks(tmp_path: Path, fake: FakeTranscriber, long_wav: Path) -> None:
    fake.delay = 0.3
    h = Harness(make_settings(tmp_path), fake)
    h.pool.start()
    job_id = h.add_job(long_wav)
    h.queue.publish(job_id)
    deadline = time.monotonic() + 5
    while not fake.calls and time.monotonic() < deadline:  # first chunk is in flight
        time.sleep(0.01)

    h.pool.stop()  # current chunk finishes, the rest is handed back
    job = h.db.get_job(job_id)
    assert job is not None and job.status == dbm.QUEUED
    assert 1 <= len(fake.calls) < 3
    assert h.db.done_chunks(job_id) == set(range(len(fake.calls)))


def test_recover_republishes_queued_and_stale_jobs(tmp_path: Path, fake: FakeTranscriber, tone_wav: Path) -> None:
    h = Harness(make_settings(tmp_path), fake)
    queued = h.add_job(tone_wav, "queued-job")
    stale = h.add_job(tone_wav, "stale-job")
    h.db.claim_job(stale, "dead-worker")
    with h.db.transaction() as conn:
        conn.execute("UPDATE jobs SET locked_at=? WHERE id=?", (time.time() - 10_000, stale))

    h.pool.recover()
    assert h.queue.size() == 2
    assert h.db.get_status(stale) == dbm.QUEUED
    h.pool.start()
    try:
        assert h.wait(queued, {dbm.DONE}).status == dbm.DONE
        assert h.wait(stale, {dbm.DONE}).attempts == 2
    finally:
        h.pool.stop()
