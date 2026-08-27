import os
from dataclasses import dataclass
from urllib.parse import urlparse


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
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_revision: str = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    model_device: str = "cpu"
    model_dtype: str = "float32"
    inference_backend: str = "token_loop"
    max_active_sequences: int = 8
    token_max_batch_size: int = 8
    token_scheduling_policy: str = "balanced"
    prefill_chunk_size_tokens: int = 0
    kv_cache_total_blocks: int = 1024
    kv_cache_block_size_tokens: int = 16
    kv_cache_bytes_per_block: int = 737_280
    kv_cache_high_watermark: float = 0.9
    kv_cache_low_watermark: float = 0.7
    prefix_cache_max_entries: int = 32
    prefix_cache_max_tokens: int = 2048
    scheduler_policy: str = "no_batching"
    max_prompt_tokens: int = 1024
    max_prompt_characters: int = 16_384
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
    api_keys: str = ""
    cors_allowed_origins: str = ""
    platform_api_enabled: bool = False
    otel_service_name: str = "llm-inference-scheduler"
    otel_exporter_otlp_endpoint: str = ""
    allow_remote_telemetry: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=_env("HOST", "0.0.0.0", str),
            port=_env("PORT", 8000, int),
            model_name=_env(
                "MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct", str
            ),
            model_revision=_env(
                "MODEL_REVISION",
                "12fd25f77366fa6b3b4b768ec3050bf629380bac",
                str,
            ),
            model_device=_env("MODEL_DEVICE", "cpu", str),
            model_dtype=_env("MODEL_DTYPE", "float32", str),
            inference_backend=_env("INFERENCE_BACKEND", "token_loop", str),
            max_active_sequences=_env("MAX_ACTIVE_SEQUENCES", 8, int),
            token_max_batch_size=_env("TOKEN_MAX_BATCH_SIZE", 8, int),
            token_scheduling_policy=_env(
                "TOKEN_SCHEDULING_POLICY", "balanced", str
            ),
            prefill_chunk_size_tokens=_env("PREFILL_CHUNK_SIZE_TOKENS", 0, int),
            kv_cache_total_blocks=_env("KV_CACHE_TOTAL_BLOCKS", 1024, int),
            kv_cache_block_size_tokens=_env(
                "KV_CACHE_BLOCK_SIZE_TOKENS", 16, int
            ),
            kv_cache_bytes_per_block=_env("KV_CACHE_BYTES_PER_BLOCK", 737_280, int),
            kv_cache_high_watermark=_env("KV_CACHE_HIGH_WATERMARK", 0.9, float),
            kv_cache_low_watermark=_env("KV_CACHE_LOW_WATERMARK", 0.7, float),
            prefix_cache_max_entries=_env("PREFIX_CACHE_MAX_ENTRIES", 32, int),
            prefix_cache_max_tokens=_env("PREFIX_CACHE_MAX_TOKENS", 2048, int),
            scheduler_policy=_env("SCHEDULER_POLICY", "no_batching", str),
            max_prompt_tokens=_env("MAX_PROMPT_TOKENS", 1024, int),
            max_prompt_characters=_env("MAX_PROMPT_CHARACTERS", 16_384, int),
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
            api_keys=_env("API_KEYS", "", str),
            cors_allowed_origins=_env("CORS_ALLOWED_ORIGINS", "", str),
            platform_api_enabled=_bool_env("PLATFORM_API_ENABLED", False),
            otel_service_name=_env(
                "OTEL_SERVICE_NAME", "llm-inference-scheduler", str
            ),
            otel_exporter_otlp_endpoint=_env(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "", str
            ),
            allow_remote_telemetry=_bool_env("ALLOW_REMOTE_TELEMETRY", False),
        )

    def validate(self) -> None:
        if self.scheduler_policy not in {"no_batching", "dynamic_batch"}:
            raise ValueError("SCHEDULER_POLICY must be no_batching or dynamic_batch")
        if self.model_device != "cpu" and not (
            self.model_device == "cuda"
            or self.model_device.startswith("cuda:")
            and self.model_device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("MODEL_DEVICE must be cpu, cuda, or cuda:<index>")
        if self.model_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("MODEL_DTYPE must be auto, float32, float16, or bfloat16")
        if self.inference_backend not in {"token_loop", "legacy_generate"}:
            raise ValueError("INFERENCE_BACKEND must be token_loop or legacy_generate")
        if self.token_scheduling_policy not in {
            "prefill_first",
            "decode_first",
            "balanced",
        }:
            raise ValueError(
                "TOKEN_SCHEDULING_POLICY must be prefill_first, decode_first, or balanced"
            )
        for name in (
            "port", "max_prompt_tokens", "max_prompt_characters", "max_new_tokens", "max_queue_size",
            "max_concurrent_requests", "max_batch_size", "max_total_batch_tokens",
            "metrics_sample_limit", "max_active_sequences",
            "token_max_batch_size", "kv_cache_total_blocks",
            "kv_cache_block_size_tokens", "kv_cache_bytes_per_block",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.token_max_batch_size > self.max_active_sequences:
            raise ValueError("TOKEN_MAX_BATCH_SIZE cannot exceed MAX_ACTIVE_SEQUENCES")
        if (self.request_timeout_seconds <= 0 or self.shutdown_grace_seconds <= 0
                or self.max_wait_ms < 0):
            raise ValueError("timeout must be positive and wait must be non-negative")
        if not (
            0
            < self.kv_cache_low_watermark
            < self.kv_cache_high_watermark
            <= 1
        ):
            raise ValueError("KV cache watermarks must satisfy 0 < low < high <= 1")
        if self.prefill_chunk_size_tokens < 0:
            raise ValueError("PREFILL_CHUNK_SIZE_TOKENS must be non-negative")
        if self.prefix_cache_max_entries < 0 or self.prefix_cache_max_tokens < 0:
            raise ValueError("prefix cache limits must be non-negative")
        if bool(self.prefix_cache_max_entries) != bool(self.prefix_cache_max_tokens):
            raise ValueError(
                "PREFIX_CACHE_MAX_ENTRIES and PREFIX_CACHE_MAX_TOKENS must both be zero or positive"
            )
        if not self.otel_service_name.strip():
            raise ValueError("OTEL_SERVICE_NAME must not be empty")
        if self.otel_exporter_otlp_endpoint:
            parsed = urlparse(self.otel_exporter_otlp_endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("OTEL_EXPORTER_OTLP_ENDPOINT must be an HTTP(S) URL")
            if (
                parsed.hostname.lower() not in {"localhost", "127.0.0.1", "::1"}
                and not self.allow_remote_telemetry
            ):
                raise ValueError(
                    "remote telemetry requires ALLOW_REMOTE_TELEMETRY=true"
                )
