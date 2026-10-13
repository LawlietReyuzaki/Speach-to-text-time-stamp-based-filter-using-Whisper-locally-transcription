"""Input validation, ffmpeg normalization to 16 kHz mono wav, and fixed-length chunking."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from app.config import ALLOWED_EXTENSIONS
from app.errors import PermanentError, TransientError


def validate_input(path: Path, max_bytes: int) -> None:
    """Cheap checks before spending CPU. Content is still verified by ffmpeg later."""
    if not path.is_file():
        raise PermanentError(f"file not found: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise PermanentError(f"unsupported extension {path.suffix!r}, allowed: {sorted(ALLOWED_EXTENSIONS)}")
    size = path.stat().st_size
    if size == 0:
        raise PermanentError("file is empty")
    if size > max_bytes:
        raise PermanentError(f"file is {size} bytes, limit is {max_bytes}")


def _run_ffmpeg(args: list[str], ffmpeg_bin: str, timeout_sec: float) -> None:
    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except FileNotFoundError as exc:
        raise TransientError(f"ffmpeg binary not found: {ffmpeg_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"ffmpeg timed out after {timeout_sec}s") from exc
    if proc.returncode != 0:
        # ffmpeg only fails here on unreadable input, which no retry will fix.
        raise PermanentError(f"ffmpeg could not decode input: {proc.stderr.strip()[-500:]}")


def normalize(src: Path, dst: Path, ffmpeg_bin: str, timeout_sec: float) -> Path:
    """Decode anything ffmpeg understands into 16 kHz mono 16-bit PCM wav."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", str(dst)],
        ffmpeg_bin,
        timeout_sec,
    )
    if wav_duration(dst) <= 0:
        raise PermanentError("input contains no audio")
    return dst


def wav_duration(path: Path) -> float:
    """Exact duration from the wav header; cheaper and more precise than ffprobe."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def split(wav: Path, out_dir: Path, chunk_sec: int, ffmpeg_bin: str, timeout_sec: float) -> list[Path]:
    """Return the chunk files in order. Short input is returned as a single chunk."""
    if wav_duration(wav) <= chunk_sec:
        return [wav]
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-i", str(wav), "-f", "segment", "-segment_time", str(chunk_sec), "-c", "copy",
         str(out_dir / "chunk_%04d.wav")],
        ffmpeg_bin,
        timeout_sec,
    )
    chunks = sorted(out_dir.glob("chunk_*.wav"))
    if not chunks:
        raise PermanentError("ffmpeg produced no chunks")
    return chunks
