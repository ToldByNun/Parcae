from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from parcae_agent import SCHEMA_ID
from parcae_agent.config import (
    AgentConfigError,
    agent_config_from_mapping,
    load_agent_config,
)

AGENTS_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = AGENTS_ROOT / "configs"
REPO_ROOT = AGENTS_ROOT.parent


def _minimal(**overrides: object) -> dict:
    raw: dict = {
        "schema": SCHEMA_ID,
        "provider": {
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key_env": None,
            "model": "llama3.1",
        },
        "parcae_bin_dir": str(REPO_ROOT / "build-cuda" / "tools" / "Debug"),
        "data_dir": str(REPO_ROOT / "data"),
        "workspace": "demo-ws",
        "allow_cuda": False,
        "budgets": {
            "max_steps": 8,
            "max_tool_calls": 16,
            "max_wall_seconds": 60,
        },
    }
    raw.update(overrides)
    return raw


def test_schema_id_constant() -> None:
    assert SCHEMA_ID == "parcae.agent_config.v0"


def test_load_ollama_example() -> None:
    cfg = load_agent_config(CONFIGS / "ollama.example.yaml")
    assert cfg.schema == SCHEMA_ID
    assert cfg.provider.base_url == "http://127.0.0.1:11434/v1"
    assert cfg.provider.api_key_env is None
    assert cfg.provider.model == "llama3.1"
    assert cfg.workspace == "_example"
    assert cfg.allow_cuda is False
    assert cfg.budgets.max_steps == 32
    assert cfg.data_dir == (REPO_ROOT / "data").resolve()
    assert cfg.provider.resolve_api_key({}) is None


def test_load_openrouter_example() -> None:
    cfg = load_agent_config(CONFIGS / "openrouter.example.yaml")
    assert cfg.provider.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.provider.base_url == "https://openrouter.ai/api/v1"
    with pytest.raises(AgentConfigError, match="OPENROUTER_API_KEY"):
        cfg.provider.resolve_api_key({})
    assert cfg.provider.resolve_api_key({"OPENROUTER_API_KEY": "sk-test"}) == "sk-test"


def test_public_dict_has_no_secret_material() -> None:
    cfg = agent_config_from_mapping(_minimal())
    public = cfg.to_public_dict()
    dumped = json.dumps(public)
    assert "sk-" not in dumped
    assert public["provider"]["api_key_env"] is None
    assert "api_key" not in public["provider"]


def test_rejects_wrong_schema() -> None:
    with pytest.raises(AgentConfigError, match="schema must be"):
        agent_config_from_mapping(_minimal(schema="parcae.agent_config.v1"))


def test_rejects_bad_workspace() -> None:
    with pytest.raises(AgentConfigError, match="workspace"):
        agent_config_from_mapping(_minimal(workspace="../evil"))
    with pytest.raises(AgentConfigError, match="workspace"):
        agent_config_from_mapping(_minimal(workspace="Evil"))


def test_rejects_data_dir_under_fixtures(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures" / "solved" / "a-warning"
    fixture_root.mkdir(parents=True)
    raw = _minimal(data_dir=str(fixture_root))
    with pytest.raises(AgentConfigError, match="fixtures"):
        agent_config_from_mapping(raw)


def test_rejects_non_http_base_url() -> None:
    raw = _minimal()
    raw["provider"] = dict(raw["provider"])
    raw["provider"]["base_url"] = "ftp://example"
    with pytest.raises(AgentConfigError, match="base_url"):
        agent_config_from_mapping(raw)


def test_rejects_non_positive_budgets() -> None:
    raw = _minimal()
    raw["budgets"] = {
        "max_steps": 0,
        "max_tool_calls": 1,
        "max_wall_seconds": 1,
    }
    with pytest.raises(AgentConfigError, match="max_steps"):
        agent_config_from_mapping(raw)


def test_rejects_unknown_top_level_field() -> None:
    with pytest.raises(AgentConfigError, match="unknown"):
        agent_config_from_mapping(_minimal(shell="bash"))


def test_relative_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    data = tmp_path / "data"
    bins = tmp_path / "bins"
    data.mkdir()
    bins.mkdir()
    cfg_path = tmp_path / "agent.yaml"
    payload = {
        "schema": SCHEMA_ID,
        "provider": {
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key_env": None,
            "model": "m",
        },
        "parcae_bin_dir": "bins",
        "data_dir": "data",
        "workspace": "ws-a",
        "budgets": {
            "max_steps": 1,
            "max_tool_calls": 1,
            "max_wall_seconds": 1,
        },
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_agent_config(cfg_path)
    assert cfg.data_dir == data.resolve()
    assert cfg.parcae_bin_dir == bins.resolve()


def test_json_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(_minimal()), encoding="utf-8")
    cfg = load_agent_config(path)
    assert cfg.workspace == "demo-ws"
