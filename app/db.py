"""SQLite persistence: schema, thread-local connections, atomic state transitions."""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from app.transcribe import Segment

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    filename    TEXT NOT NULL,
    model       TEXT NOT NULL,
    language    TEXT,
    duration    REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    worker_id   TEXT,
    locked_at   REAL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    result_of   TEXT REFERENCES jobs(id)
);
CREATE INDEX IF NOT EXISTS jobs_sha_model ON jobs(sha256, model, status);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS segments (
    job_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    seg_index   INTEGER NOT NULL,
    start       REAL NOT NULL,
    "end"       REAL NOT NULL,
    text        TEXT NOT NULL,
    PRIMARY KEY (job_id, chunk_index, seg_index)
);
"""

# queued -> processing -> done | failed | cancelled
QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class Job:
    id: str
    status: str
    storage_key: str
    sha256: str
    filename: str
    model: str
    language: str | None
    duration: float | None
    attempts: int
    error: str | None
    worker_id: str | None
    locked_at: float | None
    created_at: float
    updated_at: float
    result_of: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(**{k: row[k] for k in row.keys()})


class Database:
    """One connection per thread; every write is a short BEGIN IMMEDIATE transaction."""

    def __init__(self, path: Path, busy_timeout_ms: int = 5000) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn().executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # isolation_level=None gives autocommit; transaction() opens explicit ones.
            conn = sqlite3.connect(self.path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        # BEGIN IMMEDIATE takes the write lock up front so we never deadlock on upgrade.
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # ---- jobs -------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        storage_key: str,
        sha256: str,
        filename: str,
        model: str,
        language: str | None,
        status: str = QUEUED,
        result_of: str | None = None,
    ) -> Job:
        now = time.time()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, storage_key, sha256, filename, model, language,"
                " created_at, updated_at, result_of) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (job_id, status, storage_key, sha256, filename, model, language, now, now, result_of),
            )
        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> Job | None:
        row = self._conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def get_status(self, job_id: str) -> str | None:
        row = self._conn().execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row["status"] if row else None

    def claim_job(self, job_id: str, worker_id: str) -> bool:
        """Atomic claim: exactly one caller wins, duplicate queue messages lose."""
        now = time.time()
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, locked_at=?, worker_id=?, attempts=attempts+1,"
                " error=NULL, updated_at=? WHERE id=? AND status=?",
                (PROCESSING, now, worker_id, now, job_id, QUEUED),
            )
            return cur.rowcount == 1

    def heartbeat(self, job_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE jobs SET locked_at=? WHERE id=?", (time.time(), job_id))

    def set_language(self, job_id: str, language: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE jobs SET language=? WHERE id=?", (language, job_id))

    def finish_job(self, job_id: str, language: str, duration: float) -> bool:
        return self._transition(job_id, PROCESSING, DONE, language=language, duration=duration)

    def fail_job(self, job_id: str, error: str) -> bool:
        return self._transition(job_id, PROCESSING, FAILED, error=error)

    def cancel_job(self, job_id: str) -> bool:
        return self._transition(job_id, PROCESSING, CANCELLED)

    def requeue_job(self, job_id: str, error: str | None = None) -> bool:
        return self._transition(job_id, PROCESSING, QUEUED, error=error)

    def retry_failed_job(self, job_id: str) -> bool:
        """Manual retry: failed -> queued with attempts reset. False if not failed."""
        now = time.time()
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, attempts=0, error=NULL, worker_id=NULL,"
                " locked_at=NULL, updated_at=? WHERE id=? AND status=?",
                (QUEUED, now, job_id, FAILED),
            )
            return cur.rowcount == 1

    def _transition(
        self,
        job_id: str,
        from_status: str,
        to_status: str,
        error: str | None = None,
        language: str | None = None,
        duration: float | None = None,
    ) -> bool:
        """Guarded state change: returns False if the job already left from_status
        (for example DELETE marked it cancelled while the worker was still running)."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, error=?, language=COALESCE(?, language),"
                " duration=COALESCE(?, duration), worker_id=NULL, locked_at=NULL,"
                " updated_at=? WHERE id=? AND status=?",
                (to_status, error, language, duration, time.time(), job_id, from_status),
            )
            return cur.rowcount == 1

    def delete_job(self, job_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def find_by_hash(self, sha256: str, model: str, statuses: Sequence[str]) -> Job | None:
        """Oldest job with this content hash and model in one of the given statuses."""
        marks = ",".join("?" * len(statuses))
        row = self._conn().execute(
            f"SELECT * FROM jobs WHERE sha256=? AND model=? AND status IN ({marks})"
            " AND result_of IS NULL ORDER BY created_at LIMIT 1",
            (sha256, model, *statuses),
        ).fetchone()
        return Job.from_row(row) if row else None

    def job_ids_with_status(self, status: str, updated_before: float | None = None) -> list[str]:
        rows = self._conn().execute(
            "SELECT id FROM jobs WHERE status=? AND updated_at < ? ORDER BY created_at",
            (status, updated_before if updated_before is not None else float("inf")),
        ).fetchall()
        return [r["id"] for r in rows]

    def requeue_stale(self, stale_sec: float, max_attempts: int) -> list[str]:
        """Processing jobs whose heartbeat is older than stale_sec go back to queued.

        Jobs that already used all attempts are failed instead, so a job that keeps
        killing its worker cannot loop forever.
        """
        cutoff = time.time() - stale_sec
        now = time.time()
        with self.transaction() as conn:
            stale = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM jobs WHERE status=? AND locked_at < ? AND attempts < ?",
                    (PROCESSING, cutoff, max_attempts),
                )
            ]
            conn.execute(
                "UPDATE jobs SET status=?, error='worker lost (stale heartbeat)', worker_id=NULL,"
                " locked_at=NULL, updated_at=? WHERE status=? AND locked_at < ? AND attempts >= ?",
                (FAILED, now, PROCESSING, cutoff, max_attempts),
            )
            if stale:
                marks = ",".join("?" * len(stale))
                conn.execute(
                    f"UPDATE jobs SET status=?, worker_id=NULL, locked_at=NULL, updated_at=?"
                    f" WHERE id IN ({marks})",
                    (QUEUED, now, *stale),
                )
        return stale

    # ---- segments ---------------------------------------------------------

    def upsert_segments(self, job_id: str, chunk_index: int, segments: Sequence[Segment]) -> None:
        """Idempotent checkpoint: re-running a chunk overwrites the same rows."""
        with self.transaction() as conn:
            conn.executemany(
                'INSERT INTO segments (job_id, chunk_index, seg_index, start, "end", text)'
                " VALUES (?,?,?,?,?,?) ON CONFLICT(job_id, chunk_index, seg_index)"
                ' DO UPDATE SET start=excluded.start, "end"=excluded."end", text=excluded.text',
                [(job_id, chunk_index, i, s.start, s.end, s.text) for i, s in enumerate(segments)],
            )

    def done_chunks(self, job_id: str) -> set[int]:
        rows = self._conn().execute(
            "SELECT DISTINCT chunk_index FROM segments WHERE job_id=?", (job_id,)
        ).fetchall()
        return {r["chunk_index"] for r in rows}

    def segments(self, job_id: str) -> list[Segment]:
        rows = self._conn().execute(
            'SELECT start, "end", text FROM segments WHERE job_id=? ORDER BY chunk_index, seg_index',
            (job_id,),
        ).fetchall()
        return [Segment(r["start"], r["end"], r["text"]) for r in rows]

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
