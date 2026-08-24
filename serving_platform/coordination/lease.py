from __future__ import annotations

import uuid
from typing import Protocol


class CoordinationLease(Protocol):
    def acquire(self) -> bool: ...
    def renew(self) -> bool: ...
    def release(self) -> bool: ...
    def is_owner(self) -> bool: ...


class RedisLeaseClient(Protocol):
    def get(self, key: str) -> bytes | str | None: ...
    def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        px: int | None = None,
    ) -> object: ...
    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object: ...


RENEW_SCRIPT = """
-- coordination_renew_v1
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('PEXPIRE', KEYS[1], ARGV[2])
"""

RELEASE_SCRIPT = """
-- coordination_release_v1
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""


class RedisLease:
    """Owner-checked renewable lease for one shared coordination responsibility."""

    def __init__(
        self,
        client: RedisLeaseClient,
        name: str,
        ttl_seconds: float,
        owner_token: str | None = None,
    ):
        if not name or "{" in name or "}" in name:
            raise ValueError("lease name is invalid")
        if ttl_seconds <= 0:
            raise ValueError("lease TTL must be positive")
        self.client = client
        self.key = f"llm:coordination:{{{name}}}:lease"
        self.ttl_ms = max(1, round(ttl_seconds * 1000))
        self.owner_token = owner_token or uuid.uuid4().hex

    def acquire(self) -> bool:
        return bool(
            self.client.set(
                self.key,
                self.owner_token,
                nx=True,
                px=self.ttl_ms,
            )
        )

    def renew(self) -> bool:
        return bool(
            self.client.eval(
                RENEW_SCRIPT,
                1,
                self.key,
                self.owner_token,
                self.ttl_ms,
            )
        )

    def release(self) -> bool:
        return bool(
            self.client.eval(RELEASE_SCRIPT, 1, self.key, self.owner_token)
        )

    def is_owner(self) -> bool:
        value = self.client.get(self.key)
        if isinstance(value, bytes):
            value = value.decode()
        return value == self.owner_token
