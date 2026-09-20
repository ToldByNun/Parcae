"""Shared mock LLM helpers for CI-safe (no-network) agent tests."""

from __future__ import annotations

from typing import Any

from parcae_agent.llm import ChatCompletionResult, ChatMessage, ToolCall


class ScriptedLlm:
    """Deterministic LLM stand-in: pops scripted ChatCompletionResult values.

    Never opens sockets. Safe for CI and offline runs.
    """

    def __init__(self, script: list[ChatCompletionResult]) -> None:
        self._script = list(script)
        self.calls = 0
        self.last_messages: list[ChatMessage] = []
        self.last_tools: list[dict[str, Any]] | None = None
        self.http_posts = 0

    def chat_completions(
        self,
        messages: Any,
        *,
        tools: Any = None,
        tool_choice: Any = None,
        temperature: Any = None,
        max_tokens: Any = None,
    ) -> ChatCompletionResult:
        self.calls += 1
        self.last_messages = list(messages)
        self.last_tools = list(tools) if tools is not None else None
        if not self._script:
            raise AssertionError("ScriptedLlm script exhausted")
        return self._script.pop(0)


def assistant(
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    *,
    finish_reason: str | None = None,
    model: str = "mock",
) -> ChatCompletionResult:
    reason = finish_reason
    if reason is None:
        reason = "tool_calls" if tool_calls else "stop"
    return ChatCompletionResult(
        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
        finish_reason=reason,
        model=model,
        raw={},
    )


def tool_call(name: str, arguments: str, *, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def refuse_http_post(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    """Inject into LlmClient to prove CI tests never hit the network."""
    raise AssertionError(f"unexpected HTTP in CI-safe test: {url}")
