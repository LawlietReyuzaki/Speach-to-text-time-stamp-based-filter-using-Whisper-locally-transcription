from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient

from tests.conftest import HEADERS, FakeTranscriber, make_tone, upload, wait_for


def test_upload_poll_and_formats(make_client: Callable[..., TestClient], tone_wav: Path) -> None:
    client = make_client()
    created = upload(client, tone_wav)
    assert created["status"] == "queued"

    body = wait_for(client, created["job_id"], {"done"})
    assert body["language"] == "en"
    assert body["attempts"] == 1
    assert abs(body["duration"] - 0.8) < 0.1
    assert body["file"] == "tone.wav"
    assert body["segments"] == [{"id": 0, "start": 0.0, "end": body["duration"], "text": "text of normalized"}]
    assert body["text"] == "text of normalized"

    url = f"/v1/transcriptions/{created['job_id']}"
    srt = client.get(url, params={"format": "srt"}, headers=HEADERS).text
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,8")
    vtt = client.get(url, params={"format": "vtt"}, headers=HEADERS).text
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> ")
    txt = client.get(url, params={"format": "txt"}, headers=HEADERS).text
    assert txt == "text of normalized\n"
    assert client.get(url, params={"format": "pdf"}, headers=HEADERS).status_code == 400


def test_auth_and_error_shape(make_client: Callable[..., TestClient]) -> None:
    client = make_client()
    resp = client.get("/v1/transcriptions/nope")
    assert resp.status_code == 401
    assert resp.json() == {"error": {"code": "unauthorized", "message": "missing or invalid X-API-Key"}}
    resp = client.get("/v1/transcriptions/nope", headers=HEADERS)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_bad_uploads(make_client: Callable[..., TestClient], tmp_path: Path, tone_wav: Path) -> None:
    client = make_client(max_upload_bytes=50_000)
    resp = client.post("/v1/transcriptions", files={"file": ("notes.txt", b"hello")}, headers=HEADERS)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_input"
    resp = client.post("/v1/transcriptions", files={"file": ("empty.mp3", b"")}, headers=HEADERS)
    assert resp.status_code == 400
    resp = client.post("/v1/transcriptions", files={"file": ("big.wav", b"x" * 60_000)}, headers=HEADERS)
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "too_large"
    assert not list((tmp_path / "data").glob("*/")), "rejected uploads must not leave folders behind"

    # Garbage with a valid extension: ffmpeg rejects it, job fails permanently.
    resp = client.post("/v1/transcriptions", files={"file": ("bad.mp3", b"not audio" * 10)}, headers=HEADERS)
    assert resp.status_code == 202
    body = wait_for(client, resp.json()["job_id"], {"failed"})
    assert body["attempts"] == 1 and "ffmpeg" in body["error"]


def test_queue_full_returns_429(make_client: Callable[..., TestClient], tmp_path: Path) -> None:
    client = make_client(workers=0, queue_size=1)
    first = make_tone(tmp_path / "a.wav", 0.3)
    second = make_tone(tmp_path / "b.wav", 0.4)
    assert upload(client, first)["status"] == "queued"

    with second.open("rb") as fh:
        resp = client.post("/v1/transcriptions", files={"file": ("b.wav", fh)}, headers=HEADERS)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "30"
    assert resp.json()["error"]["code"] == "queue_full"
    assert client.get(f"/v1/transcriptions/{resp.json().get('job_id')}", headers=HEADERS).status_code == 404
    assert len(list((tmp_path / "data").glob("*/"))) == 1, "rejected upload was deleted from disk"
    health = client.get("/health").json()
    assert health["queue_size"] == 1 and health["active_workers"] == 0 and health["workers"] == 0
    assert health["model"] == {"name": "whisper-fake", "device": "cpu", "compute_type": "int8"}


def test_duplicate_upload_reuses_transcript(make_client: Callable[..., TestClient], fake: FakeTranscriber, tone_wav: Path) -> None:
    client = make_client()
    first = upload(client, tone_wav)
    done = wait_for(client, first["job_id"], {"done"})

    second = upload(client, tone_wav)
    assert second["status"] == "done" and second["job_id"] != first["job_id"]
    body = client.get(f"/v1/transcriptions/{second['job_id']}", headers=HEADERS).json()
    assert body["result_of"] == first["job_id"]
    assert body["segments"] == done["segments"]
    assert len(fake.calls) == 1, "second upload must not transcribe again"


def test_duplicate_upload_in_flight_returns_existing_job(make_client: Callable[..., TestClient], tone_wav: Path) -> None:
    client = make_client(workers=0)
    first = upload(client, tone_wav)
    second = upload(client, tone_wav)
    assert second == {"job_id": first["job_id"], "status": "queued"}


def test_delete_while_processing_cancels(make_client: Callable[..., TestClient], fake: FakeTranscriber, long_wav: Path, tmp_path: Path) -> None:
    fake.delay = 0.3
    client = make_client()
    job_id = upload(client, long_wav)["job_id"]
    wait_for(client, job_id, {"processing"})

    resp = client.delete(f"/v1/transcriptions/{job_id}", headers=HEADERS)
    assert resp.status_code == 202 and resp.json()["status"] == "cancelled"
    wait_for(client, job_id, {"deleted"})
    assert len(fake.calls) < 3, "worker stopped between chunks"
    assert not (tmp_path / "data" / job_id).exists()


def test_delete_finished_job(make_client: Callable[..., TestClient], tone_wav: Path, tmp_path: Path) -> None:
    client = make_client()
    job_id = upload(client, tone_wav)["job_id"]
    wait_for(client, job_id, {"done"})
    assert client.delete(f"/v1/transcriptions/{job_id}", headers=HEADERS).status_code == 204
    assert client.get(f"/v1/transcriptions/{job_id}", headers=HEADERS).status_code == 404
    assert not (tmp_path / "data" / job_id).exists()


def test_manual_retry_of_failed_job(make_client: Callable[..., TestClient], fake: FakeTranscriber, tone_wav: Path) -> None:
    from app.errors import TransientError

    fake.fail = lambda n: TransientError("flaky")
    client = make_client(max_attempts=1)
    job_id = upload(client, tone_wav)["job_id"]
    assert wait_for(client, job_id, {"failed"})["error"] == "TransientError: flaky"

    fake.fail = lambda n: None
    resp = client.post(f"/v1/transcriptions/{job_id}/retry", headers=HEADERS)
    assert resp.status_code == 202
    assert wait_for(client, job_id, {"done"})["attempts"] == 1
    assert client.post(f"/v1/transcriptions/{job_id}/retry", headers=HEADERS).status_code == 409


def test_rate_limit(make_client: Callable[..., TestClient], tmp_path: Path) -> None:
    client = make_client(workers=0, rate_limit_per_min=2)
    for i in range(2):
        upload(client, make_tone(tmp_path / f"t{i}.wav", 0.2 + i / 10))
    with (tmp_path / "t0.wav").open("rb") as fh:
        resp = client.post("/v1/transcriptions", files={"file": ("t0.wav", fh)}, headers=HEADERS)
    assert resp.status_code == 429 and resp.json()["error"]["code"] == "rate_limited"


def test_health_reports_active_workers(make_client: Callable[..., TestClient], fake: FakeTranscriber, long_wav: Path) -> None:
    fake.delay = 0.2
    client = make_client(workers=2)
    job_id = upload(client, long_wav)["job_id"]
    wait_for(client, job_id, {"processing"})
    time.sleep(0.05)
    assert client.get("/health").json()["active_workers"] == 1
    wait_for(client, job_id, {"done"})


def test_index_serves_ui(make_client: Callable[..., TestClient]) -> None:
    resp = make_client().get("/")
    assert resp.status_code == 200 and "<title>Transcription</title>" in resp.text


def test_jwt_flow(make_client: Callable[..., TestClient], tone_wav: Path) -> None:
    from app import auth

    client = make_client(workers=0)
    assert client.post("/v1/auth/token").status_code == 401
    token = client.post("/v1/auth/token", headers=HEADERS).json()
    assert token["token_type"] == "bearer" and token["expires_in"] == 60
    bearer = {"Authorization": f"Bearer {token['access_token']}"}

    with tone_wav.open("rb") as fh:
        resp = client.post("/v1/transcriptions", files={"file": ("tone.wav", fh)}, headers=bearer)
    assert resp.status_code == 202
    assert client.get(f"/v1/transcriptions/{resp.json()['job_id']}", headers=bearer).status_code == 200

    tampered = {"Authorization": "Bearer " + token["access_token"][:-3] + "abc"}
    assert client.get("/v1/transcriptions/x", headers=tampered).json()["error"]["message"] == "bad signature"
    expired = auth.create_token("test-secret", "api-key", ttl_sec=-1)
    resp = client.get("/v1/transcriptions/x", headers={"Authorization": f"Bearer {expired}"})
    assert resp.status_code == 401 and resp.json()["error"]["message"] == "token expired"
    wrong_secret = auth.create_token("other", "api-key", ttl_sec=60)
    assert client.get("/v1/transcriptions/x", headers={"Authorization": f"Bearer {wrong_secret}"}).status_code == 401


def test_languages_and_audio_download(make_client: Callable[..., TestClient], tone_wav: Path) -> None:
    client = make_client(workers=0)
    langs = client.get("/v1/languages").json()
    assert langs["languages"]["ur"] == "Urdu" and len(langs["languages"]) == 100

    with tone_wav.open("rb") as fh:
        resp = client.post("/v1/transcriptions", files={"file": ("tone.wav", fh)}, data={"language": "xx"}, headers=HEADERS)
    assert resp.status_code == 400 and "unknown language" in resp.json()["error"]["message"]

    job_id = upload(client, tone_wav, language="en")["job_id"]
    resp = client.get(f"/v1/transcriptions/{job_id}/audio", headers=HEADERS)
    assert resp.status_code == 200 and resp.content == tone_wav.read_bytes()
