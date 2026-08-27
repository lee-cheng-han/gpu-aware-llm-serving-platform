from dataclasses import replace

import pytest

from apps.gateway.config import Settings


def test_settings_parse_warmup(monkeypatch):
    monkeypatch.setenv("MODEL_WARMUP_ON_START", "yes")
    assert Settings.from_env().model_warmup_on_start is True


def test_environment_defaults_use_pinned_reference_model(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_REVISION", raising=False)
    settings = Settings.from_env()
    assert settings.model_name == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert settings.model_revision == "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def test_invalid_warmup_value_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_WARMUP_ON_START", "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        Settings.from_env()


def test_invalid_scheduler_policy_rejected():
    with pytest.raises(ValueError, match="SCHEDULER_POLICY"):
        replace(Settings(), scheduler_policy="unknown").validate()


def test_shutdown_and_metrics_settings_parse(monkeypatch):
    monkeypatch.setenv("SHUTDOWN_GRACE_SECONDS", "12.5")
    monkeypatch.setenv("METRICS_SAMPLE_LIMIT", "500")
    settings = Settings.from_env()
    assert settings.shutdown_grace_seconds == 12.5
    assert settings.metrics_sample_limit == 500


def test_model_device_supports_explicit_cuda_index(monkeypatch):
    monkeypatch.setenv("MODEL_DEVICE", "cuda:1")
    assert Settings.from_env().model_device == "cuda:1"
    with pytest.raises(ValueError, match="MODEL_DEVICE"):
        replace(Settings(), model_device="gpu").validate()


def test_platform_api_is_opt_in(monkeypatch):
    assert Settings.from_env().platform_api_enabled is False
    monkeypatch.setenv("PLATFORM_API_ENABLED", "true")
    assert Settings.from_env().platform_api_enabled is True


def test_token_backend_configuration_is_explicit_and_validated(monkeypatch):
    monkeypatch.setenv("MODEL_DTYPE", "bfloat16")
    monkeypatch.setenv("INFERENCE_BACKEND", "legacy_generate")
    monkeypatch.setenv("MAX_ACTIVE_SEQUENCES", "2")
    settings = Settings.from_env()
    assert settings.model_dtype == "bfloat16"
    assert settings.inference_backend == "legacy_generate"
    assert settings.max_active_sequences == 2
    with pytest.raises(ValueError, match="MODEL_DTYPE"):
        replace(Settings(), model_dtype="int8").validate()
    with pytest.raises(ValueError, match="INFERENCE_BACKEND"):
        replace(Settings(), inference_backend="unknown").validate()


def test_continuous_scheduler_and_kv_configuration(monkeypatch):
    monkeypatch.setenv("MAX_ACTIVE_SEQUENCES", "12")
    monkeypatch.setenv("TOKEN_MAX_BATCH_SIZE", "6")
    monkeypatch.setenv("TOKEN_SCHEDULING_POLICY", "decode_first")
    monkeypatch.setenv("PREFILL_CHUNK_SIZE_TOKENS", "128")
    monkeypatch.setenv("KV_CACHE_TOTAL_BLOCKS", "256")
    settings = Settings.from_env()
    assert settings.max_active_sequences == 12
    assert settings.token_max_batch_size == 6
    assert settings.token_scheduling_policy == "decode_first"
    assert settings.prefill_chunk_size_tokens == 128
    assert settings.kv_cache_total_blocks == 256

    with pytest.raises(ValueError, match="TOKEN_SCHEDULING_POLICY"):
        replace(Settings(), token_scheduling_policy="random").validate()
    with pytest.raises(ValueError, match="watermarks"):
        replace(
            Settings(),
            kv_cache_low_watermark=0.95,
            kv_cache_high_watermark=0.9,
        ).validate()
    with pytest.raises(ValueError, match="TOKEN_MAX_BATCH_SIZE"):
        replace(
            Settings(), max_active_sequences=2, token_max_batch_size=3
        ).validate()


def test_prefix_cache_can_be_bounded_or_disabled(monkeypatch):
    monkeypatch.setenv("PREFIX_CACHE_MAX_ENTRIES", "8")
    monkeypatch.setenv("PREFIX_CACHE_MAX_TOKENS", "512")
    settings = Settings.from_env()
    assert settings.prefix_cache_max_entries == 8
    assert settings.prefix_cache_max_tokens == 512
    replace(settings, prefix_cache_max_entries=0, prefix_cache_max_tokens=0).validate()
    with pytest.raises(ValueError, match="must both be zero or positive"):
        replace(settings, prefix_cache_max_entries=0).validate()


def test_telemetry_defaults_off_and_remote_export_requires_opt_in(monkeypatch):
    settings = Settings.from_env()
    assert settings.otel_exporter_otlp_endpoint == ""
    replace(
        settings,
        otel_exporter_otlp_endpoint="http://localhost:4318/v1/traces",
    ).validate()
    with pytest.raises(ValueError, match="remote telemetry"):
        replace(
            settings,
            otel_exporter_otlp_endpoint="https://telemetry.example/v1/traces",
        ).validate()
    replace(
        settings,
        otel_exporter_otlp_endpoint="https://telemetry.example/v1/traces",
        allow_remote_telemetry=True,
    ).validate()
