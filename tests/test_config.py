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
