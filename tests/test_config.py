from dataclasses import replace

import pytest

from app.config import Settings


def test_settings_parse_warmup(monkeypatch):
    monkeypatch.setenv("MODEL_WARMUP_ON_START", "yes")
    assert Settings.from_env().model_warmup_on_start is True


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
