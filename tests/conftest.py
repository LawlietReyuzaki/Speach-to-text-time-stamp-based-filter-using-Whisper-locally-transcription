from __future__ import annotations

import os
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.audio import wav_duration
from app.config import Settings
from app.main import create_app
from app.transcribe import ChunkResult, Segment

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}


def make_tone(path: Path, seconds: float, *codec: str) -> Path:
    """A sine tone is enough: ffmpeg must decode it, the model is faked anyway."""
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}", "-ar", "44100", "-ac", "2", *codec, str(path)],
        check=True,
    )
    return path


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = dict(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "jobs.db",
        api_key=API_KEY,
        jwt_secret="test-secret",
        jwt_ttl_sec=60,
        model_size="fake",
        compute_type="int8",
        device="cpu",
        workers=1,
        share_model=False,
        queue_size=8,
        chunk_sec=1,
        max_upload_bytes=2_000_000,
        max_attempts=3,
        retry_base_sec=0.0,
        stale_sec=300,
        sweep_interval_sec=3600,
        ffmpeg_bin=FFMPEG,
        ffmpeg_timeout_sec=60,
        chunk_timeout_sec=60,
        rate_limit_per_min=1000,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class FakeTranscriber:
    """Returns one segment spanning the chunk; `fail(call_number)` can inject errors."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail: Callable[[int], Exception | None] = lambda n: None
        self.delay = 0.0
        self._lock = threading.Lock()

    def transcribe(self, wav: Path, language: str | None) -> ChunkResult:
        with self._lock:
            self.calls.append(wav.name)
            call_number = len(self.calls)
        exc = self.fail(call_number)
        if exc is not None:
            raise exc
        time.sleep(self.delay)
        return ChunkResult(language or "en", [Segment(0.0, wav_duration(wav), f"text of {wav.stem}")])


@pytest.fixture(scope="session")
def tone_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tone(tmp_path_factory.mktemp("audio") / "tone.wav", 0.8)


@pytest.fixture(scope="session")
def long_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # 2.5 s with chunk_sec=1 gives chunks of 1, 1 and 0.5 seconds.
    return make_tone(tmp_path_factory.mktemp("audio") / "long.wav", 2.5)


@pytest.fixture
def fake() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def make_client(tmp_path: Path, fake: FakeTranscriber) -> Iterator[Callable[..., TestClient]]:
    """Factory so a test can tweak settings (workers=0, queue_size=1, ...)."""
    with ExitStack() as stack:

        def _make(**overrides: object) -> TestClient:
            app = create_app(make_settings(tmp_path, **overrides), lambda lock: fake)
            return stack.enter_context(TestClient(app))

        yield _make


def upload(client: TestClient, path: Path, **form: str) -> dict:
    with path.open("rb") as fh:
        resp = client.post("/v1/transcriptions", files={"file": (path.name, fh)}, data=form, headers=HEADERS)
    assert resp.status_code in (200, 202), resp.text
    return resp.json()


def wait_for(client: TestClient, job_id: str, statuses: set[str], timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/transcriptions/{job_id}", headers=HEADERS)
        if resp.status_code == 404 and "deleted" in statuses:
            return {"status": "deleted"}
        body = resp.json()
        if body["status"] in statuses:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {statuses} in {timeout}s")
