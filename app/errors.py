"""Error classes that decide how the worker reacts to a failure."""


class PermanentError(Exception):
    """The job can never succeed (corrupt file, no audio, unsupported). No retry."""


class TransientError(Exception):
    """The job might succeed later (timeout, I/O, OOM, unexpected). Retried with backoff."""


class JobInterrupted(Exception):
    """The job was cancelled or the process is shutting down; stop between chunks."""
