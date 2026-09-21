"""CI-safe agent loop integration (mock LLM, no network, no real CLIs).

Normative agent-tooling commit 30: `parcae-agent run` path must succeed offline with
a scripted model and a fake ToolBridge runner.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from parcae_agent.cli import cmd_run
from parcae_agent.config import agent_config_from_mapping
from parcae_agent.loop import AgentLoop, AgentStopReason
from parcae_agent.tool_bridge import TOOL_RESPONSE_SCHEMA, SubprocessResult, ToolBridge
from parcae_agent.tool_schemas import openai_tools
from parcae_agent.transcript import load_transcript
from tests.mocking import ScriptedLlm, assistant, refuse_http_post, tool_call

pytestmark = pytest.mark.ci


def _config(tmp_path: Path, **overrides: object):
    raw = {
        "schema": "parcae.agent_config.v0",
        "provider": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key_env": None,
            "model": "mock-ci",
        },
        "parcae_bin_dir": str(tmp_path / "bins"),
        "data_dir": str(tmp_path / "data"),
        "workspace": "ci-mock-ws",
        "allow_cuda": False,
        "budgets": {
            "max_steps": 6,
            "max_tool_calls": 12,
            "max_wall_seconds": 30,
        },
    }
    raw.update(overrides)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bins").mkdir(parents=True, exist_ok=True)
    return agent_config_from_mapping(raw)


def _envelope(tool: str, *, ok: bool = True, result: dict | None = None) -> dict:
    return {
        "schema": TOOL_RESPONSE_SCHEMA,
        "ok": ok,
        "tool": tool,
        "backend": "cpu",
        "result": result if result is not None else {},
        "error": None
        if ok
        else {"code": "validation", "message": "failed"},
    }


def _runner_from_tool_name():
    """Map argv to a canned envelope by guessing the tool from flags/binary."""

    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        joined = " ".join(argv)
        if "parcae-catalog" in joined or "parcae-catalog.exe" in joined:
            tool = "catalog"
            result = {"transforms": ["atbash", "caesar"]}
        elif "set-status" in argv or "parcae-hypothesis" in joined and "set-status" in joined:
            tool = "hypothesis_set_status"
            result = {"id": "h-ci", "status": "promoted"}
        elif "parcae-validate" in joined:
            tool = "validate"
            result = {"fixture_id": "a-warning", "passed": True}
        else:
            tool = "unknown"
            result = {}
        body = json.dumps(_envelope(tool, result=result))
        return SubprocessResult(0, body, "")

    return runner


def test_ci_mock_loop_catalog_then_validate_succeeds(tmp_path: Path) -> None:
    """Full loop: catalog → validate(ok) → stop with Succeeded; transcript on disk."""
    cfg = _config(tmp_path)
    llm = ScriptedLlm(
        [
            assistant(
                tool_calls=(
                    tool_call("catalog", '{"transforms":true}', call_id="c1"),
                )
            ),
            assistant(
                tool_calls=(
                    tool_call(
                        "validate",
                        '{"id":"a-warning","require_locked":true}',
                        call_id="v1",
                    ),
                )
            ),
            assistant("should not be reached"),
        ]
    )
    bridge = ToolBridge(cfg, runner=_runner_from_tool_name())
    from parcae_agent.transcript import TranscriptWriter

    writer = TranscriptWriter(
        cfg,
        run_id="c1c1c1c1",
        utc_now=lambda: "2026-09-20T03:00:00Z",
    )
    result = AgentLoop(cfg, llm, bridge, transcript=writer).run(  # type: ignore[arg-type]
        "Validate a-warning after checking the catalog."
    )

    assert result.stop_reason == AgentStopReason.Succeeded
    assert result.exit_code == 0
    assert result.ok
    assert result.tool_calls == 2
    assert llm.calls == 2  # stopped on validate success before third turn
    assert llm.last_tools is not None
    schema_names = {t["function"]["name"] for t in openai_tools()}
    assert {t["function"]["name"] for t in llm.last_tools} == schema_names

    steps = load_transcript(writer.path)
    roles = [s["role"] for s in steps]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles.count("tool") == 2
    assert roles[-1] == "finish"
    assert steps[-1]["ok"] is True
    tools = [s["tool"] for s in steps if s["role"] == "tool"]
    assert tools == ["catalog", "validate"]


def test_ci_mock_cli_run_end_to_end(tmp_path: Path) -> None:
    """`cmd_run` with injected mock LLM + fake bridge — no sockets, no binaries."""
    cfg = _config(tmp_path)
    # Write config file for cmd_run path.
    import yaml

    cfg_path = tmp_path / "agent.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema": "parcae.agent_config.v0",
                "provider": {
                    "base_url": "http://127.0.0.1:9/v1",
                    "api_key_env": None,
                    "model": "mock-ci",
                },
                "parcae_bin_dir": str(tmp_path / "bins"),
                "data_dir": str(tmp_path / "data"),
                "workspace": "ci-cli-ws",
                "allow_cuda": False,
                "budgets": {
                    "max_steps": 4,
                    "max_tool_calls": 8,
                    "max_wall_seconds": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    llm = ScriptedLlm(
        [
            assistant(
                tool_calls=(
                    tool_call(
                        "hypothesis_set_status",
                        '{"id":"h-ci","status":"promoted"}',
                        call_id="h1",
                    ),
                )
            )
        ]
    )
    out, err = io.StringIO(), io.StringIO()
    code = cmd_run(
        cfg_path,
        prompt="Promote the working hypothesis.",
        as_json=True,
        persist_transcript=True,
        llm_factory=lambda _c: llm,  # type: ignore[arg-type,return-value]
        bridge_factory=lambda c: ToolBridge(c, runner=_runner_from_tool_name()),
        stdout=out,
        stderr=err,
    )
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["stop_reason"] == "succeeded"
    assert payload["tool_calls"] == 1
    assert payload["run_id"]
    assert payload["transcript_path"]
    assert Path(payload["transcript_path"]).is_file()


def test_ci_llm_client_never_calls_http_when_scripted(tmp_path: Path) -> None:
    """Even constructing LlmClient with refuse_http_post must not be hit by ScriptedLlm."""
    from parcae_agent.llm import LlmClient
    from parcae_agent.config import ProviderConfig

    # Guard: if someone swaps ScriptedLlm for a real client in this module, fail loud.
    client = LlmClient(
        ProviderConfig(base_url="http://127.0.0.1:9/v1", model="x"),
        http_post=refuse_http_post,
    )
    with pytest.raises(AssertionError, match="unexpected HTTP"):
        client.chat_completions([{"role": "user", "content": "ping"}])

    # Scripted path still works without touching http_post.
    llm = ScriptedLlm([assistant("pong")])
    cfg = _config(tmp_path)
    result = AgentLoop(
        cfg,
        llm,  # type: ignore[arg-type]
        ToolBridge(cfg, runner=_runner_from_tool_name()),
    ).run("ping")
    assert result.stop_reason == AgentStopReason.Completed
    assert result.final_text == "pong"
