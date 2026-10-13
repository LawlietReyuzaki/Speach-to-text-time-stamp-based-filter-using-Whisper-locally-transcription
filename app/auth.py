"""Minimal HS256 JWT (stdlib only). Clients trade the API key for a short-lived token."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


class InvalidToken(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def create_token(secret: str, subject: str, ttl_sec: int) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    payload = _b64(json.dumps({"sub": subject, "iat": now, "exp": now + ttl_sec}).encode())
    signature = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64(signature)}"


def verify_token(secret: str, token: str) -> dict[str, Any]:
    """Return the claims, or raise InvalidToken on bad signature, shape, or expiry."""
    try:
        header, payload, signature = token.split(".")
    except ValueError as exc:
        raise InvalidToken("malformed token") from exc
    expected = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _unb64(signature)):
        raise InvalidToken("bad signature")
    claims = json.loads(_unb64(payload))
    if claims.get("exp", 0) < time.time():
        raise InvalidToken("token expired")
    return claims
