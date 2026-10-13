"""faster-whisper wrapper. Returns segments with timestamps relative to the given file."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.errors import TransientError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ChunkResult:
    language: str
    segments: list[Segment]


# Sentence enders across scripts: Latin . ! ?, Urdu/Arabic full stop ۔ and question mark ؟,
# Devanagari danda ।, CJK 。！？ .
SENTENCE_END = tuple(".!?\u06d4\u061f\u0964\u3002\uff01\uff1f")


def sentences_from_words(words: list[Word]) -> list[Segment]:
    """Group word timestamps into sentences: each runs from its first word's start
    to its last word's end. Whisper's own segments are phrases, not sentences."""
    sentences: list[Segment] = []
    current: list[Word] = []
    for word in words:
        if not word.text.strip():
            continue
        current.append(word)
        if word.text.rstrip().endswith(SENTENCE_END):
            sentences.append(_join(current))
            current = []
    if current:
        sentences.append(_join(current))
    return sentences


def _join(words: list[Word]) -> Segment:
    text = " ".join(w.text.strip() for w in words)
    return Segment(words[0].start, words[-1].end, text)


class Transcriber(Protocol):
    def transcribe(self, wav: Path, language: str | None) -> ChunkResult:
        """Transcribe one 16 kHz mono wav. language=None means auto-detect."""


class WhisperTranscriber:
    """Loads the model once. Pass a lock to share one instance between worker threads."""

    def __init__(
        self,
        model_size: str,
        compute_type: str = "int8",
        device: str = "cpu",
        timeout_sec: float = 900,
        lock: threading.Lock | None = None,
    ) -> None:
        if device == "cuda":
            _register_cuda_dlls()
        from faster_whisper import WhisperModel  # imported lazily: slow and heavy

        log.info("loading whisper model=%s device=%s compute=%s", model_size, device, compute_type)
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._timeout = timeout_sec
        self._lock = lock or threading.Lock()

    def transcribe(self, wav: Path, language: str | None) -> ChunkResult:
        with self._lock:
            return self._run(wav, language)

    def _run(self, wav: Path, language: str | None) -> ChunkResult:
        started = time.monotonic()
        raw, info = self._model.transcribe(
            str(wav), language=language, vad_filter=True, word_timestamps=True
        )
        words: list[Word] = []
        # The generator decodes lazily, so this loop is where the time goes and where
        # we can check the clock. A hard kill would require running in a subprocess.
        for seg in raw:
            if time.monotonic() - started > self._timeout:
                raise TransientError(f"transcription exceeded {self._timeout}s for {wav.name}")
            if seg.words:
                words.extend(Word(float(w.start), float(w.end), w.word) for w in seg.words)
            elif seg.text.strip():  # no word alignment available: keep the phrase as one unit
                words.append(Word(float(seg.start), float(seg.end), seg.text.strip()))
        return ChunkResult(language=info.language, segments=sentences_from_words(words))


def _register_cuda_dlls() -> None:
    """On Windows the pip packages nvidia-cublas-cu12 / nvidia-cudnn-cu12 put their DLLs in
    site-packages/nvidia/*/bin. ctranslate2 loads them by name via PATH, so extend PATH."""
    if sys.platform != "win32":
        return
    bins = [
        os.path.join(base, "nvidia", name, "bin")
        for base in sys.path
        if os.path.isdir(os.path.join(base, "nvidia"))
        for name in os.listdir(os.path.join(base, "nvidia"))
        if os.path.isdir(os.path.join(base, "nvidia", name, "bin"))
    ]
    os.environ["PATH"] = os.pathsep.join([*bins, os.environ.get("PATH", "")])
