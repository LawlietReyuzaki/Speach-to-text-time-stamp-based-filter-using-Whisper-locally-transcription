"""Transcript output formats: json dict, srt, vtt, txt."""
from __future__ import annotations

from typing import Any, Sequence

from app.transcribe import Segment


def to_json(filename: str, language: str | None, duration: float, segments: Sequence[Segment]) -> dict[str, Any]:
    items = [
        {"id": i, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
        for i, s in enumerate(segments)
        if s.text.strip()
    ]
    return {
        "file": filename,
        "language": language,
        "duration": round(duration, 2),
        "segments": items,
        "text": " ".join(item["text"] for item in items),
    }


def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: Sequence[Segment]) -> str:
    blocks = [
        f"{i}\n{_timestamp(s.start, ',')} --> {_timestamp(s.end, ',')}\n{s.text}\n"
        for i, s in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(segments: Sequence[Segment]) -> str:
    blocks = [f"{_timestamp(s.start, '.')} --> {_timestamp(s.end, '.')}\n{s.text}\n" for s in segments]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_txt(segments: Sequence[Segment]) -> str:
    return "\n".join(s.text for s in segments) + "\n"
