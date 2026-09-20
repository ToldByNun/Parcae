"""Always-on gates: live smoke stays off in CI unless explicitly enabled."""

from __future__ import annotations

import os

from tests.test_provider_smoke import live_enabled, openrouter_key_present


def test_live_gate_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("PARCAE_AGENT_LIVE", raising=False)
    assert live_enabled() is False
    monkeypatch.setenv("PARCAE_AGENT_LIVE", "1")
    assert live_enabled() is True
    monkeypatch.setenv("PARCAE_AGENT_LIVE", "yes")
    assert live_enabled() is True


def test_openrouter_key_helper(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert openrouter_key_present() is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert openrouter_key_present() is True
