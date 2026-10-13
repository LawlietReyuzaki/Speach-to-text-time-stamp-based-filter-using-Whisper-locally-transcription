from __future__ import annotations

import threading
import time
from pathlib import Path

from app import db as dbm
from app.db import Database
from app.transcribe import Segment


def _db(tmp_path: Path) -> Database:
    return Database(tmp_path / "jobs.db")


def _job(db: Database, job_id: str = "job1") -> str:
    db.create_job(job_id, f"{job_id}/original.wav", "abc", "a.wav", "small", None)
    return job_id


def test_atomic_claim_only_one_thread_wins(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job_id = _job(db)
    barrier = threading.Barrier(2)
    wins: list[bool] = []

    def claim(worker: str) -> None:
        barrier.wait()
        wins.append(db.claim_job(job_id, worker))

    threads = [threading.Thread(target=claim, args=(f"w{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(wins) == [False, True]
    job = db.get_job(job_id)
    assert job is not None and job.status == dbm.PROCESSING and job.attempts == 1


def test_stale_heartbeat_is_requeued_by_sweeper(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job_id = _job(db)
    assert db.claim_job(job_id, "w1")
    with db.transaction() as conn:  # simulate a worker that died 10 minutes ago
        conn.execute("UPDATE jobs SET locked_at=? WHERE id=?", (time.time() - 600, job_id))

    assert db.requeue_stale(stale_sec=300, max_attempts=3) == [job_id]
    job = db.get_job(job_id)
    assert job is not None and job.status == dbm.QUEUED and job.worker_id is None
    # A second claim counts the next attempt.
    assert db.claim_job(job_id, "w2")
    assert db.get_job(job_id).attempts == 2  # type: ignore[union-attr]


def test_stale_job_out_of_attempts_is_failed(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job_id = _job(db)
    assert db.claim_job(job_id, "w1")
    with db.transaction() as conn:
        conn.execute("UPDATE jobs SET locked_at=?, attempts=3 WHERE id=?", (time.time() - 600, job_id))

    assert db.requeue_stale(stale_sec=300, max_attempts=3) == []
    assert db.get_status(job_id) == dbm.FAILED


def test_fresh_heartbeat_is_left_alone(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job_id = _job(db)
    db.claim_job(job_id, "w1")
    db.heartbeat(job_id)
    assert db.requeue_stale(stale_sec=300, max_attempts=3) == []
    assert db.get_status(job_id) == dbm.PROCESSING


def test_upsert_segments_is_idempotent(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job_id = _job(db)
    segs = [Segment(0.0, 1.0, "a"), Segment(1.0, 2.0, "b")]
    db.upsert_segments(job_id, 0, segs)
    db.upsert_segments(job_id, 0, segs)
    db.upsert_segments(job_id, 1, [Segment(2.0, 3.0, "c")])

    assert db.segments(job_id) == segs + [Segment(2.0, 3.0, "c")]
    assert db.done_chunks(job_id) == {0, 1}


def test_transitions_are_guarded(tmp_path: Path) -> None:
    db = _db(tmp_path)
    job_id = _job(db)
    db.claim_job(job_id, "w1")
    assert db.cancel_job(job_id)
    # The worker finishing late must not overwrite the cancellation.
    assert not db.finish_job(job_id, "en", 1.0)
    assert db.get_status(job_id) == dbm.CANCELLED
