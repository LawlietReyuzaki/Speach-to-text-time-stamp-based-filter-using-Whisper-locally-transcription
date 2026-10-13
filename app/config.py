"""Environment based settings. Every knob mentioned in the README lives here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"})
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    api_key: str
    jwt_secret: str
    jwt_ttl_sec: int
    model_size: str
    compute_type: str
    device: str
    workers: int
    share_model: bool
    queue_size: int
    chunk_sec: int
    max_upload_bytes: int
    max_attempts: int
    retry_base_sec: float
    stale_sec: int
    sweep_interval_sec: int
    ffmpeg_bin: str
    ffmpeg_timeout_sec: int
    chunk_timeout_sec: int
    rate_limit_per_min: int


def load_settings() -> Settings:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    return Settings(
        data_dir=data_dir,
        db_path=Path(os.environ.get("DB_PATH", str(data_dir / "jobs.db"))),
        api_key=os.environ.get("API_KEY", "dev-key"),
        jwt_secret=os.environ.get("JWT_SECRET", "change-me"),
        jwt_ttl_sec=_env_int("JWT_TTL_SEC", 3600),
        model_size=os.environ.get("MODEL_SIZE", "small"),
        compute_type=os.environ.get("COMPUTE_TYPE", "int8"),
        device=os.environ.get("DEVICE", "cpu"),
        workers=_env_int("WORKERS", 2),
        share_model=_env_bool("SHARE_MODEL", False),
        queue_size=_env_int("QUEUE_SIZE", 100),
        chunk_sec=_env_int("CHUNK_SEC", 600),
        max_upload_bytes=_env_int("MAX_UPLOAD_MB", 200) * 1024 * 1024,
        max_attempts=_env_int("MAX_ATTEMPTS", 3),
        retry_base_sec=float(os.environ.get("RETRY_BASE_SEC", "2")),
        stale_sec=_env_int("STALE_SEC", 300),
        sweep_interval_sec=_env_int("SWEEP_SEC", 30),
        ffmpeg_bin=os.environ.get("FFMPEG_BIN", "ffmpeg"),
        ffmpeg_timeout_sec=_env_int("FFMPEG_TIMEOUT_SEC", 300),
        chunk_timeout_sec=_env_int("CHUNK_TIMEOUT_SEC", 900),
        rate_limit_per_min=_env_int("RATE_LIMIT_PER_MIN", 10),
    )
