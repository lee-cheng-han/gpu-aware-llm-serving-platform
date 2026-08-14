from __future__ import annotations

import hmac
from dataclasses import dataclass
from hashlib import sha256

from fastapi import Request

from apps.gateway.errors import APIError


@dataclass(frozen=True)
class TenantIdentity:
    tenant_id: str


class ApiKeyAuthenticator:
    """Development authentication; secrets are compared but never returned or logged."""

    def __init__(self, credentials: dict[str, str] | None = None):
        credentials = dict(credentials or {})
        if any(not tenant or not key for key, tenant in credentials.items()):
            raise ValueError("API keys and tenant identifiers must be non-empty")
        self._credentials = tuple(
            (sha256(key.encode()).digest(), tenant_id)
            for key, tenant_id in credentials.items()
        )

    @property
    def enabled(self) -> bool:
        return bool(self._credentials)

    def authenticate(self, request: Request) -> TenantIdentity:
        if not self.enabled:
            return TenantIdentity("default")
        authorization = request.headers.get("authorization", "")
        scheme, _, supplied = authorization.partition(" ")
        if scheme.lower() != "bearer" or not supplied:
            raise APIError(401, "authentication_required", "a bearer API key is required")
        supplied_digest = sha256(supplied.encode()).digest()
        for key_digest, tenant_id in self._credentials:
            if hmac.compare_digest(supplied_digest, key_digest):
                return TenantIdentity(tenant_id)
        raise APIError(401, "invalid_api_key", "the API key is invalid")


def parse_api_keys(value: str) -> dict[str, str]:
    """Parse `tenant:key,tenant:key`; returned mapping is secret -> tenant."""
    if not value.strip():
        return {}
    credentials: dict[str, str] = {}
    for entry in value.split(","):
        tenant_id, separator, key = entry.partition(":")
        if not separator or not tenant_id.strip() or not key.strip():
            raise ValueError("API_KEYS entries must use tenant:key")
        if key.strip() in credentials:
            raise ValueError("API keys must be unique")
        credentials[key.strip()] = tenant_id.strip()
    return credentials
