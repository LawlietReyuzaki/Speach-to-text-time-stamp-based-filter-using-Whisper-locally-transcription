"""Standalone transcription: python cli.py file.mp3 -> prints the transcript as JSON."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app import audio, formats
from app.config import load_settings
from app.errors import PermanentError, TransientError
from app.transcribe import WhisperTranscriber
from app.worker import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe one audio file to JSON.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--language", default=None, help="ISO code, e.g. en. Default: auto-detect")
    args = parser.parse_args(argv)
    settings = load_settings()

    try:
        audio.validate_input(args.file, settings.max_upload_bytes)
        transcriber = WhisperTranscriber(
            settings.model_size, settings.compute_type, settings.device, settings.chunk_timeout_sec
        )
        with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
            result = run_pipeline(args.file, Path(tmp), transcriber, settings, language=args.language)
    except (PermanentError, TransientError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = formats.to_json(args.file.name, result.language, result.duration, result.segments)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
