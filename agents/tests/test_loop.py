from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from parcae_agent.config import agent_config_from_mapping
from parcae_agent.llm import ChatCompletionResult, ChatMessage, ToolCall
from parcae_agent.loop import AgentLoop, AgentStopReason
from parcae_agent.prompts import LIBER_PRIMUS_SYSTEM_PROMPT, liber_primus_system_prompt
from parcae_agent.tool_bridge import SubprocessResult, ToolBridge, TOOL_RESPONSE_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides: object):
    raw: dict[str, Any] = {
        "schema": "parcae.agent_config.v0",
        "provider": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key_env": None,
            "model": "mock",
        },
        "parcae_bin_dir": str(REPO_ROOT / "build-cuda" / "tools" / "Debug"),
        "data_dir": str(REPO_ROOT / "data"),
        "workspace": "loop-ws",
        "allow_cuda": False,
        "budgets": {
            "max_steps": 8,
            "max_tool_calls": 16,
            "max_wall_seconds": 60,
        },
    }
    raw.update(overrides)
    return agent_config_from_mapping(raw)


def _ok_envelope(tool: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": TOOL_RESPONSE_SCHEMA,
        "ok": True,
        "tool": tool,
        "backend": "cpu",
        "result": result or {},
        "error": None,
    }


class ScriptedLlm:
    """Minimal stand-in matching LlmClient.chat_completions."""

    def __init__(self, script: list[ChatCompletionResult]) -> None:
        self._script = list(script)
        self.calls = 0
        self.last_messages: list[ChatMessage] = []
        self.last_tools: list[dict[str, Any]] | None = None

    def chat_completions(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
    ) -> ChatCompletionResult:
        self.calls += 1
        self.last_messages = list(messages)
        self.last_tools = list(tools) if tools is not None else None
        if not self._script:
            raise AssertionError("ScriptedLlm exhausted")
        return self._script.pop(0)


def _assistant(
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    finish_reason: str = "stop",
) -> ChatCompletionResult:
    return ChatCompletionResult(
        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
        finish_reason=finish_reason if not tool_calls else "tool_calls",
        model="mock",
        raw={},
    )


def test_system_prompt_mentions_catalog_and_no_shell() -> None:
    text = liber_primus_system_prompt(workspace="ws-a", allow_cuda=False)
    assert "catalog" in text.lower()
    assert "shell" in text.lower()
    assert "ws-a" in text
    assert "allow_cuda: false" in text
    assert "Z_29" in LIBER_PRIMUS_SYSTEM_PROMPT or "Z_29" in text


def test_loop_completes_without_tools() -> None:
    llm = ScriptedLlm([_assistant("Nothing to do.")])
    cfg = _config()
    bridge = ToolBridge(cfg, runner=lambda *a, **k: SubprocessResult(0, "{}", ""))
    loop = AgentLoop(cfg, llm, bridge)  # type: ignore[arg-type]
    result = loop.run("ping")
    assert result.stop_reason == AgentStopReason.Completed
    assert result.ok
    assert result.exit_code == 0
    assert result.final_text == "Nothing to do."
    assert result.tool_calls == 0
    assert result.messages[0].role == "system"
    assert result.messages[1].role == "user"
    assert "catalog" in (result.messages[0].content or "").lower()
    assert llm.last_tools is not None
    assert any(t["function"]["name"] == "catalog" for t in llm.last_tools)


def test_loop_runs_tool_then_stops_on_text() -> None:
    envelopes = {
        "catalog": _ok_envelope("catalog", {"transforms": ["atbash"]}),
    }

    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        # argv contains binary + flags; tool name is not in argv — detect via flags.
        tool = "catalog"
        return SubprocessResult(0, json.dumps(envelopes[tool]), "")

    llm = ScriptedLlm(
        [
            _assistant(
                content=None,
                tool_calls=(
                    ToolCall(id="c1", name="catalog", arguments='{"transforms":true}'),
                ),
            ),
            _assistant("Catalog looks good."),
        ]
    )
    cfg = _config()
    bridge = ToolBridge(cfg, runner=runner)
    result = AgentLoop(cfg, llm, bridge).run("list transforms")  # type: ignore[arg-type]
    assert result.stop_reason == AgentStopReason.Completed
    assert result.tool_calls == 1
    assert result.steps[0].tool_results[0].ok
    assert result.messages[-1].content == "Catalog looks good."
    # Tool result fed back as role=tool
    assert any(m.role == "tool" for m in result.messages)


def test_loop_succeeds_on_validate_ok() -> None:
    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        return SubprocessResult(0, json.dumps(_ok_envelope("validate")), "")

    llm = ScriptedLlm(
        [
            _assistant(
                tool_calls=(
                    ToolCall(
                        id="v1",
                        name="validate",
                        arguments='{"id":"a-warning","require_locked":true}',
                    ),
                )
            ),
            # Would be next turn — must not be consumed.
            _assistant("should not run"),
        ]
    )
    cfg = _config()
    result = AgentLoop(cfg, llm, ToolBridge(cfg, runner=runner)).run("validate")  # type: ignore[arg-type]
    assert result.stop_reason == AgentStopReason.Succeeded
    assert result.exit_code == 0
    assert llm.calls == 1
    assert result.tool_calls == 1


def test_loop_succeeds_on_hypothesis_promoted() -> None:
    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        return SubprocessResult(
            0, json.dumps(_ok_envelope("hypothesis_set_status")), ""
        )

    llm = ScriptedLlm(
        [
            _assistant(
                tool_calls=(
                    ToolCall(
                        id="h1",
                        name="hypothesis_set_status",
                        arguments='{"id":"h-1","status":"promoted"}',
                    ),
                )
            )
        ]
    )
    cfg = _config()
    result = AgentLoop(cfg, llm, ToolBridge(cfg, runner=runner)).run("promote")  # type: ignore[arg-type]
    assert result.stop_reason == AgentStopReason.Succeeded


def test_loop_budget_steps() -> None:
    # Always ask for another tool — never stop with text.
    def forever(_messages, **_kwargs):
        return _assistant(
            tool_calls=(
                ToolCall(id="c", name="catalog", arguments='{"transforms":true}'),
            )
        )

    class AlwaysTools(ScriptedLlm):
        def chat_completions(self, messages, **kwargs):
            self.calls += 1
            self.last_messages = list(messages)
            return forever(messages, **kwargs)

    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        return SubprocessResult(0, json.dumps(_ok_envelope("catalog")), "")

    cfg = _config(budgets={"max_steps": 2, "max_tool_calls": 99, "max_wall_seconds": 60})
    result = AgentLoop(cfg, AlwaysTools([]), ToolBridge(cfg, runner=runner)).run("x")  # type: ignore[arg-type]
    assert result.stop_reason == AgentStopReason.BudgetSteps
    assert result.exit_code == 1
    assert result.tool_calls == 2


def test_loop_budget_tool_calls() -> None:
    llm = ScriptedLlm(
        [
            _assistant(
                tool_calls=(
                    ToolCall(id="a", name="catalog", arguments="{}"),
                    ToolCall(id="b", name="catalog", arguments="{}"),
                )
            )
        ]
    )

    def runner(argv: list[str], timeout: float | None) -> SubprocessResult:
        raise AssertionError("should not run — budget blocks before invoke")

    cfg = _config(budgets={"max_steps": 5, "max_tool_calls": 1, "max_wall_seconds": 60})
    result = AgentLoop(cfg, llm, ToolBridge(cfg, runner=runner)).run("x")  # type: ignore[arg-type]
    assert result.stop_reason == AgentStopReason.BudgetToolCalls
    assert result.exit_code == 1
    assert result.tool_calls == 0


def test_loop_budget_wall() -> None:
    ticks = {"n": 0}

    def clock() -> float:
        # First call = start (0), then each check advances past budget.
        ticks["n"] += 1
        return 0.0 if ticks["n"] == 1 else 100.0

    llm = ScriptedLlm([_assistant("hi")])
    cfg = _config(budgets={"max_steps": 5, "max_tool_calls": 5, "max_wall_seconds": 1})
    bridge = ToolBridge(cfg, runner=lambda *a, **k: SubprocessResult(0, "{}", ""))
    result = AgentLoop(cfg, llm, bridge, clock=clock).run("x")  # type: ignore[arg-type]
    assert result.stop_reason == AgentStopReason.BudgetWall
    assert result.exit_code == 1


def test_loop_llm_error() -> None:
    class BoomDuck:
        def chat_completions(self, *args, **kwargs):
            from parcae_agent.llm import LlmError

            raise LlmError("network down")

    cfg = _config()
    bridge = ToolBridge(cfg, runner=lambda *a, **k: SubprocessResult(0, "{}", ""))
    result = AgentLoop(cfg, BoomDuck(), bridge).run("x")  # type: ignore[arg-type]
    assert result.stop_reason == AgentStopReason.Error
    assert result.exit_code == 2
    assert "network down" in (result.error or "")
