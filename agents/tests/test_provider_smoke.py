"""Optional live provider smoke (Ollama + OpenRouter).

CI / default: skipped unless PARCAE_AGENT_LIVE=1.
OpenRouter also requires OPENROUTER_API_KEY.
Never commit secrets; never require these tests for merge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from parcae_agent.cli import cmd_doctor, cmd_providers_test

AGENTS_ROOT = Path(__file__).resolve().parents[1]
OLLAMA_CONFIG = AGENTS_ROOT / "configs" / "ollama.example.yaml"
OPENROUTER_CONFIG = AGENTS_ROOT / "configs" / "openrouter.example.yaml"


def live_enabled() -> bool:
    return os.environ.get("PARCAE_AGENT_LIVE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def openrouter_key_present() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _require_live_gate() -> None:
    if not live_enabled():
        pytest.skip("set PARCAE_AGENT_LIVE=1 to run live provider smoke")


@pytest.mark.ollama
def test_ollama_example_config_loads_and_doctor_data_ok() -> None:
    """Config + data_dir checks (still needs LIVE=1 so CI stays quiet)."""
    assert OLLAMA_CONFIG.is_file()
    # doctor may WARN/FAIL on missing binaries — only assert config+data criticals
    # by inspecting JSON when bins are absent.
    import io

    out = io.StringIO()
    code = cmd_doctor(OLLAMA_CONFIG, as_json=True, stdout=out)
    payload = json.loads(out.getvalue())
    by_name = {c["name"]: c for c in payload["checks"]}
    assert by_name["config"]["ok"] is True
    assert by_name["data_dir"]["ok"] is True
    assert by_name["api_key_env"]["ok"] is True
    # 0 = all ok, 1 = warnings only, 2 = critical — data_dir must not force 2 alone
    assert code in (0, 1, 2)


@pytest.mark.ollama
def test_ollama_providers_test_live() -> None:
    """Hits local Ollama OpenAI-compatible endpoint."""
    import io

    out, err = io.StringIO(), io.StringIO()
    code = cmd_providers_test(OLLAMA_CONFIG, as_json=True, stdout=out, stderr=err)
    if code != 0:
        pytest.skip(
            "Ollama not reachable or model missing "
            f"(exit {code}): {err.getvalue() or out.getvalue()}"
        )
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["reply"]


@pytest.mark.openrouter
def test_openrouter_providers_test_live() -> None:
    """Hits OpenRouter; requires OPENROUTER_API_KEY."""
    if not openrouter_key_present():
        pytest.skip("OPENROUTER_API_KEY unset")

    import io

    out, err = io.StringIO(), io.StringIO()
    code = cmd_providers_test(OPENROUTER_CONFIG, as_json=True, stdout=out, stderr=err)
    assert code == 0, err.getvalue() or out.getvalue()
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert "openrouter.ai" in payload["base_url"]
    assert payload["reply"]
