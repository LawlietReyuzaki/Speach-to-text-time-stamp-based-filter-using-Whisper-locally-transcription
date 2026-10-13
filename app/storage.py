"""Blob storage behind a small interface so local disk can be swapped for S3/GCS."""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class Storage(ABC):
    """Keys look like "{job_id}/original.mp3"; all files of a job share the job_id prefix."""

    @abstractmethod
    def write(self, key: str, chunks: Iterable[bytes]) -> int:
        """Write a stream of byte chunks under key and return the total bytes written."""

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """A path ffmpeg can read. A remote backend would download to a temp file here."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Remove everything stored under prefix (the whole job folder)."""


class LocalStorage(Storage):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, key: str, chunks: Iterable[bytes]) -> int:
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with path.open("wb") as fh:
            for chunk in chunks:
                fh.write(chunk)
                written += len(chunk)
        return written

    def local_path(self, key: str) -> Path:
        return self.root / key

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self.root / prefix, ignore_errors=True)
