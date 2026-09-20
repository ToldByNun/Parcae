from __future__ import annotations

import json
from pathlib import Path

from parcae_agent.config import agent_config_from_mapping
from parcae_agent.llm import ChatCompletionResult, ChatMessage, ToolCall
from parcae_agent.loop import AgentLoop, AgentStopReason
from parcae_agent.tool_bridge import (
    TOOL_RESPONSE_SCHEMA,
    SubprocessResult,
    ToolBridge,
    ToolInvocationResult,
)
from parcae_agent.transcript import (
    TRANSCRIPT_SCHEMA,
    TranscriptWriter,
    extract_hypothesis_id,
    load_transcript,
    redact_summary,
)


def _config(tmp_path: Path, **overrides: object):
    raw = {
        "schema": "parcae.agent_config.v0",
        "provider": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key_env": None,
            "model": "mock",
        },
        "parcae_bin_dir": str(tmp_path / "bins"),
        "data_dir": str(tmp_path / "data"),
        "workspace": "persist-ws",
        "allow_cuda": False,
        "budgets": {
            "max_steps": 4,
            "max_tool_calls": 8,
            "max_wall_seconds": 30,
        },
    }
    raw.update(overrides)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "bins").mkdir(parents=True, exist_ok=True)
    return agent_config_from_mapping(raw)


def test_redact_summary_strips_secrets_and_truncates() -> None:
    text = "Authorization: Bearer sk-abcdefghijklmnop and more " + ("x" * 300)
    out = redact_summary(text, max_len=80)
    assert "sk-abcdefghijklmnop" not in out
    assert "[redacted]" in out
    assert len(out) <= 80


def test_extract_hypothesis_id() -> None:
    assert (
        extract_hypothesis_id("hypothesis_init", '{"id":"h-1"}') == "h-1"
    )
    assert extract_hypothesis_id("catalog", '{"id":"h-1"}') is None
    assert (
        extract_hypothesis_id(
            "hypothesis_show",
            {},
            {"result": {"id": "h-from-result"}},
        )
        == "h-from-result"
    )


def test_writer_creates_jsonl_under_workspace(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    ticks = {"n": 0}

    def utc() -> str:
        ticks["n"] += 1
        return f"2026-09-20T00:00:0{ticks['n']}Z"

    writer = TranscriptWriter(cfg, run_id="abcd1234", save_envelopes=False, utc_now=utc)
    writer.append("system", summary="sys")
    writer.append("user", summary="hello sk-abcdefghijklmnop")
    writer.append("finish", summary="completed", ok=True)

    assert writer.path.is_file()
    assert writer.path.parent.name == "transcripts"
    assert "abcd1234" in writer.path.name
    steps = load_transcript(writer.path)
    assert len(steps) == 3
    assert steps[0]["schema"] == TRANSCRIPT_SCHEMA
    assert steps[0]["workspace_id"] == "persist-ws"
    assert steps[0]["run_id"] == "abcd1234"
    assert steps[0]["seq"] == 0
    assert steps[1]["role"] == "user"
    assert "sk-abcdefghijklmnop" not in steps[1]["summary"]
    assert steps[2]["role"] == "finish"
    assert steps[2]["ok"] is True


def test_append_tool_saves_envelope_and_hypothesis_id(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    writer = TranscriptWriter(cfg, run_id="deadbeef", utc_now=lambda: "2026-09-20T12:00:00Z")
    envelope = {
        "schema": TOOL_RESPONSE_SCHEMA,
        "ok": True,
        "tool": "hypothesis_init",
        "backend": None,
        "result": {"id": "h-9"},
        "error": None,
    }
    inv = ToolInvocationResult(
        tool="hypothesis_init",
        argv=("parcae-hypothesis",),
        returncode=0,
        envelope=envelope,
        stdout=json.dumps(envelope),
        stderr="",
    )
    step = writer.append_tool(inv, arguments='{"id":"h-9","title":"t"}')
    assert step.hypothesis_id == "h-9"
    assert step.envelope_ref is not None
    assert step.ok is True
    env_path = writer.workspace_root / step.envelope_ref
    assert env_path.is_file()
    saved = json.loads(env_path.read_text(encoding="utf-8"))
    assert saved["tool"] == "hypothesis_init"


def test_loop_persists_transcript_with_tools(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    writer = TranscriptWriter(
        cfg,
        run_id="loop0001",
        utc_now=lambda: "2026-09-20T12:00:00Z",
    )

    class Scripted:
        def __init__(self) -> None:
            self.n = 0

        def chat_completions(self, messages, **kwargs):
            self.n += 1
            if self.n == 1:
                return ChatCompletionResult(
                    message=ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=(
                            ToolCall(
                                id="c1",
                                name="catalog",
                                arguments='{"transforms":true}',
                            ),
                        ),
                    ),
                    finish_reason="tool_calls",
                    model="m",
                    raw={},
                )
            return ChatCompletionResult(
                message=ChatMessage(role="assistant", content="all done"),
                finish_reason="stop",
                model="m",
                raw={},
            )

    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        env = {
            "schema": TOOL_RESPONSE_SCHEMA,
            "ok": True,
            "tool": "catalog",
            "backend": None,
            "result": {"transforms": ["atbash"]},
            "error": None,
        }
        return SubprocessResult(0, json.dumps(env), "")

    result = AgentLoop(
        cfg,
        Scripted(),  # type: ignore[arg-type]
        ToolBridge(cfg, runner=runner),
        transcript=writer,
    ).run("list transforms")

    assert result.stop_reason == AgentStopReason.Completed
    assert result.run_id == "loop0001"
    assert result.transcript_path == writer.path
    steps = load_transcript(writer.path)
    roles = [s["role"] for s in steps]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "assistant" in roles
    assert "tool" in roles
    assert roles[-1] == "finish"
    tool_steps = [s for s in steps if s["role"] == "tool"]
    assert tool_steps[0]["tool"] == "catalog"
    assert tool_steps[0]["ok"] is True
