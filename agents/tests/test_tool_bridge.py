from __future__ import annotations

import json
from pathlib import Path

import pytest

from parcae_agent.allowlist import ALLOWED_TOOLS, DENIED_BINARIES
from parcae_agent.config import agent_config_from_mapping
from parcae_agent.llm import ToolCall
from parcae_agent.tool_bridge import (
    TOOL_RESPONSE_SCHEMA,
    SubprocessResult,
    ToolBridge,
    ToolBridgeError,
    parse_tool_response,
    synthesize_envelope,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides: object):
    raw = {
        "schema": "parcae.agent_config.v0",
        "provider": {
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key_env": None,
            "model": "m",
        },
        "parcae_bin_dir": str(REPO_ROOT / "build-cuda" / "tools" / "Debug"),
        "data_dir": str(REPO_ROOT / "data"),
        "workspace": "demo-ws",
        "allow_cuda": False,
        "budgets": {
            "max_steps": 4,
            "max_tool_calls": 8,
            "max_wall_seconds": 30,
        },
    }
    raw.update(overrides)
    return agent_config_from_mapping(raw)


def test_allowlist_covers_spec_tools() -> None:
    assert "catalog" in ALLOWED_TOOLS
    assert "hypothesis_set_status" in ALLOWED_TOOLS
    assert "search_cycle" in ALLOWED_TOOLS
    assert "parcae-blind-crack" in DENIED_BINARIES
    assert "parcae-bench" in DENIED_BINARIES
    assert "bench" in DENIED_BINARIES
    assert "parcae-search-run" in DENIED_BINARIES
    assert "search-run" in DENIED_BINARIES


def test_build_argv_search_cycle_status_only() -> None:
    bridge = ToolBridge(_config())
    argv = bridge.build_argv("search_cycle", {"status": True})
    assert argv[0].endswith("parcae-search-cycle") or argv[0].endswith(
        "parcae-search-cycle.exe"
    )
    assert "--status" in argv
    assert "--json" in argv
    assert "--data-dir" in argv
    assert "--workspace" not in argv
    assert "--omit-timing" not in argv
    assert "--quiet" not in argv
    assert "--family" not in argv


def test_build_argv_search_cycle_family_injects_workspace_omit_timing_and_quiet() -> None:
    bridge = ToolBridge(_config())
    argv = bridge.build_argv(
        "search_cycle",
        {
            "family": "atbash",
            "k": 3,
            "seed": 1,
            "iterations": 2,
            "created_utc": "2026-09-22T20:00:01Z",
            "backend": "cpu",
        },
    )
    assert "--workspace" in argv
    assert argv[argv.index("--workspace") + 1] == "demo-ws"
    assert "--omit-timing" in argv
    assert "--quiet" in argv
    assert argv[argv.index("--family") + 1] == "atbash"
    assert argv[argv.index("--k") + 1] == "3"
    assert argv[argv.index("--seed") + 1] == "1"
    assert argv[argv.index("--iterations") + 1] == "2"
    assert argv[argv.index("--created-utc") + 1] == "2026-09-22T20:00:01Z"
    assert argv[argv.index("--backend") + 1] == "cpu"
    assert "--status" not in argv


def test_build_argv_search_cycle_allow_theory_uri_flag() -> None:
    bridge = ToolBridge(_config())
    argv = bridge.build_argv(
        "search_cycle",
        {
            "job": "jobs/theory.json",
            "allow_theory_uri": True,
            "backend": "cpu",
        },
    )
    assert "--allow-theory-uri" in argv
    assert argv[argv.index("--job") + 1] == "jobs/theory.json"


def test_build_argv_search_cycle_rejects_incomplete_and_status_mix() -> None:
    bridge = ToolBridge(_config())
    with pytest.raises(ToolBridgeError, match="family or job"):
        bridge.build_argv("search_cycle", {})
    with pytest.raises(ToolBridgeError, match="status=true"):
        bridge.build_argv("search_cycle", {"status": True, "family": "caesar"})
    with pytest.raises(ToolBridgeError, match="allow_cuda"):
        bridge.build_argv(
            "search_cycle",
            {"family": "caesar", "k": 1, "backend": "cuda"},
        )


def test_build_argv_catalog_injects_data_dir_and_json() -> None:
    bridge = ToolBridge(_config())
    argv = bridge.build_argv("catalog", {"transforms": True})
    assert argv[0].endswith("parcae-catalog") or argv[0].endswith("parcae-catalog.exe")
    assert "--json" in argv
    assert "--data-dir" in argv
    data_idx = argv.index("--data-dir")
    assert argv[data_idx + 1] == str((_config().data_dir))
    assert "--transforms" in argv
    assert "shell" not in argv


def test_build_argv_rejects_bridge_owned_and_unknown_keys() -> None:
    bridge = ToolBridge(_config())
    with pytest.raises(ToolBridgeError, match="bridge-owned"):
        bridge.build_argv("catalog", {"data_dir": "/evil"})
    with pytest.raises(ToolBridgeError, match="bridge-owned"):
        bridge.build_argv("catalog", {"shell": "bash"})
    with pytest.raises(ToolBridgeError, match="unknown"):
        bridge.build_argv("catalog", {"not_a_real_flag": True})
    with pytest.raises(ToolBridgeError, match="allow-list"):
        bridge.build_argv("blind_crack", {})


def test_build_argv_hypothesis_injects_workspace_subcommand() -> None:
    bridge = ToolBridge(_config())
    argv = bridge.build_argv("hypothesis_show", {"id": "h-1"})
    assert "show" in argv
    assert argv[argv.index("--workspace") + 1] == "demo-ws"
    assert argv[argv.index("--id") + 1] == "h-1"


def test_build_argv_tokenize_positional() -> None:
    bridge = ToolBridge(_config())
    argv = bridge.build_argv("tokenize", {"input": "-", "strict": False})
    assert "--no-strict" in argv
    assert argv[-1] == "-"


def test_cuda_requires_allow_cuda_config() -> None:
    bridge = ToolBridge(_config(allow_cuda=False))
    with pytest.raises(ToolBridgeError, match="allow_cuda"):
        bridge.build_argv("score", {"score_id": "ic_mod29", "input": "-", "backend": "cuda"})

    bridge_ok = ToolBridge(_config(allow_cuda=True))
    argv = bridge_ok.build_argv(
        "score",
        {
            "score_id": "ic_mod29",
            "input": "-",
            "indices": True,
            "backend": "cuda",
        },
    )
    assert "--allow-cuda" in argv
    assert argv[argv.index("--backend") + 1] == "cuda"


def test_params_json_accepts_object_or_string() -> None:
    bridge = ToolBridge(_config())
    argv_obj = bridge.build_argv(
        "decode",
        {
            "input": "x.txt",
            "transform_id": "caesar",
            "params_json": {"shift": 3},
        },
    )
    assert json.loads(argv_obj[argv_obj.index("--params-json") + 1]) == {"shift": 3}

    argv_str = bridge.build_argv(
        "decode",
        {
            "input": "x.txt",
            "transform_id": "atbash",
            "params_json": '{"shift":1}',
        },
    )
    assert argv_str[argv_str.index("--params-json") + 1] == '{"shift":1}'


def test_invoke_policy_denial_without_running() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        calls.append(argv)
        return SubprocessResult(0, "{}", "")

    bridge = ToolBridge(_config(), runner=runner)
    result = bridge.invoke("not_a_tool", {})
    assert calls == []
    assert result.returncode == 2
    assert result.ok is False
    assert result.envelope["error"]["code"] == "policy"


def test_invoke_parses_envelope_from_runner() -> None:
    envelope = {
        "schema": TOOL_RESPONSE_SCHEMA,
        "ok": True,
        "tool": "catalog",
        "backend": None,
        "result": {"transforms": ["atbash"]},
        "error": None,
    }

    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        assert "--json" in argv
        return SubprocessResult(0, json.dumps(envelope), "")

    bridge = ToolBridge(_config(), runner=runner)
    result = bridge.invoke_tool_call(
        ToolCall(id="c1", name="catalog", arguments='{"transforms":true}')
    )
    assert result.ok is True
    assert result.envelope["result"]["transforms"] == ["atbash"]
    assert result.returncode == 0


def test_parse_tool_response_synthesizes_internal_on_garbage() -> None:
    env = parse_tool_response("not-json", tool="score")
    assert env["ok"] is False
    assert env["error"]["code"] == "internal"
    assert env["schema"] == TOOL_RESPONSE_SCHEMA


def test_synthesize_envelope_shape() -> None:
    env = synthesize_envelope("rank", code="policy", message="nope")
    assert env == {
        "schema": TOOL_RESPONSE_SCHEMA,
        "ok": False,
        "tool": "rank",
        "backend": None,
        "result": None,
        "error": {"code": "policy", "message": "nope"},
    }


def test_never_uses_shell_metacharacter_passthrough_as_command() -> None:
    """Arguments are discrete argv tokens; a fake 'command' key is rejected."""
    bridge = ToolBridge(_config())
    with pytest.raises(ToolBridgeError, match="bridge-owned|unknown"):
        bridge.build_argv("catalog", {"command": "rm -rf /"})


@pytest.mark.skipif(
    not (
        (REPO_ROOT / "build-cuda" / "tools" / "Debug" / "parcae-catalog.exe").is_file()
        or (REPO_ROOT / "build-cuda" / "tools" / "Debug" / "parcae-catalog").is_file()
        or (REPO_ROOT / "build" / "tools" / "Release" / "parcae-catalog.exe").is_file()
        or (REPO_ROOT / "build" / "tools" / "parcae-catalog").is_file()
    ),
    reason="parcae-catalog binary not built",
)
def test_live_catalog_subprocess() -> None:
    bin_dir = REPO_ROOT / "build-cuda" / "tools" / "Debug"
    if not (bin_dir / "parcae-catalog.exe").is_file() and not (
        bin_dir / "parcae-catalog"
    ).is_file():
        for candidate in (
            REPO_ROOT / "build" / "tools" / "Release",
            REPO_ROOT / "build" / "tools",
        ):
            if (candidate / "parcae-catalog.exe").is_file() or (
                candidate / "parcae-catalog"
            ).is_file():
                bin_dir = candidate
                break

    bridge = ToolBridge(_config(parcae_bin_dir=str(bin_dir)), timeout_seconds=30)
    result = bridge.invoke("catalog", {"transforms": True})
    assert result.envelope["schema"] == TOOL_RESPONSE_SCHEMA
    assert result.envelope["tool"] == "catalog"
    assert result.ok is True
    assert result.returncode == 0
