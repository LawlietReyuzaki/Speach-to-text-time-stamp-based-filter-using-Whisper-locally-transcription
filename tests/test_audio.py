from __future__ import annotations

import wave
from pathlib import Path

import pytest

from app import audio
from app.errors import PermanentError
from app.worker import run_pipeline
from tests.conftest import FFMPEG, FakeTranscriber, make_settings, make_tone


@pytest.mark.parametrize("ext,codec", [("wav", []), ("mp3", ["-c:a", "libmp3lame"]), ("m4a", ["-c:a", "aac"])])
def test_normalize_to_16k_mono_wav(tmp_path: Path, ext: str, codec: list[str]) -> None:
    src = make_tone(tmp_path / f"tone.{ext}", 0.5, *codec)
    out = audio.normalize(src, tmp_path / "out.wav", FFMPEG, 60)
    with wave.open(str(out), "rb") as wf:
        assert (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) == (16000, 1, 2)
    assert abs(audio.wav_duration(out) - 0.5) < 0.1


def test_validate_rejects_bad_inputs(tmp_path: Path, tone_wav: Path) -> None:
    with pytest.raises(PermanentError, match="not found"):
        audio.validate_input(tmp_path / "missing.wav", 10**6)
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")
    with pytest.raises(PermanentError, match="empty"):
        audio.validate_input(empty, 10**6)
    with pytest.raises(PermanentError, match="unsupported"):
        audio.validate_input(Path(__file__), 10**6)
    with pytest.raises(PermanentError, match="limit"):
        audio.validate_input(tone_wav, 10)


def test_corrupt_file_is_permanent_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is not audio" * 100)
    with pytest.raises(PermanentError, match="ffmpeg"):
        audio.normalize(bad, tmp_path / "out.wav", FFMPEG, 60)


def test_long_file_chunk_offsets(tmp_path: Path, long_wav: Path, fake: FakeTranscriber) -> None:
    settings = make_settings(tmp_path, chunk_sec=1)
    result = run_pipeline(long_wav, tmp_path / "work", fake, settings)

    assert len(fake.calls) == 3
    assert abs(result.duration - 2.5) < 0.05
    starts = [s.start for s in result.segments]
    ends = [s.end for s in result.segments]
    assert starts == pytest.approx([0.0, 1.0, 2.0], abs=0.1)
    assert ends == pytest.approx([1.0, 2.0, 2.5], abs=0.1)
    # Offsets add up to the real total, not 3 * CHUNK_SEC.
    assert ends[-1] == pytest.approx(result.duration, abs=0.01)
