import os
from dataclasses import dataclass


def _env(name: str, default, cast):
    value = os.getenv(name)
    return default if value is None else cast(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "sshleifer/tiny-gpt2"
    model_revision: str = "main"
    scheduler_policy: str = "no_batching"
    max_prompt_tokens: int = 1024
    max_new_tokens: int = 128
    max_queue_size: int = 128
    max_concurrent_requests: int = 16
    request_timeout_seconds: float = 60
    max_batch_size: int = 8
    max_wait_ms: int = 25
    max_total_batch_tokens: int = 1024
    model_warmup_on_start: bool = False
    shutdown_grace_seconds: float = 30
    metrics_sample_limit: int = 10_000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=_env("HOST", "0.0.0.0", str),
            port=_env("PORT", 8000, int),
            model_name=_env("MODEL_NAME", "sshleifer/tiny-gpt2", str),
            model_revision=_env("MODEL_REVISION", "main", str),
            scheduler_policy=_env("SCHEDULER_POLICY", "no_batching", str),
            max_prompt_tokens=_env("MAX_PROMPT_TOKENS", 1024, int),
            max_new_tokens=_env("MAX_NEW_TOKENS", 128, int),
            max_queue_size=_env("MAX_QUEUE_SIZE", 128, int),
            max_concurrent_requests=_env("MAX_CONCURRENT_REQUESTS", 16, int),
            request_timeout_seconds=_env("REQUEST_TIMEOUT_SECONDS", 60, float),
            max_batch_size=_env("MAX_BATCH_SIZE", 8, int),
            max_wait_ms=_env("MAX_WAIT_MS", 25, int),
            max_total_batch_tokens=_env("MAX_TOTAL_BATCH_TOKENS", 1024, int),
            model_warmup_on_start=_bool_env("MODEL_WARMUP_ON_START", False),
            shutdown_grace_seconds=_env("SHUTDOWN_GRACE_SECONDS", 30, float),
            metrics_sample_limit=_env("METRICS_SAMPLE_LIMIT", 10_000, int),
        )

    def validate(self) -> None:
        if self.scheduler_policy not in {"no_batching", "dynamic_batch"}:
            raise ValueError("SCHEDULER_POLICY must be no_batching or dynamic_batch")
        for name in (
            "port", "max_prompt_tokens", "max_new_tokens", "max_queue_size",
            "max_concurrent_requests", "max_batch_size", "max_total_batch_tokens",
            "metrics_sample_limit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (self.request_timeout_seconds <= 0 or self.shutdown_grace_seconds <= 0
                or self.max_wait_ms < 0):
            raise ValueError("timeout must be positive and wait must be non-negative")
