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
