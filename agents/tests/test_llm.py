from __future__ import annotations

import json
from typing import Any

import pytest

from parcae_agent.config import ProviderConfig, agent_config_from_mapping
from parcae_agent.llm import (
    ChatMessage,
    LlmClient,
    LlmError,
    ToolCall,
)


def _provider(**overrides: Any) -> ProviderConfig:
    raw = {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "llama3.1",
        "api_key_env": None,
    }
    raw.update(overrides)
    return ProviderConfig(
        base_url=raw["base_url"],
        model=raw["model"],
        api_key_env=raw["api_key_env"],
    )


def _assistant_payload(
    *,
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "llama3.1",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }


def test_chat_completions_url_joins_base() -> None:
    client = LlmClient(_provider(base_url="http://127.0.0.1:11434/v1"))
    assert client.chat_completions_url == "http://127.0.0.1:11434/v1/chat/completions"


def test_local_request_has_no_authorization() -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(body.decode("utf-8"))
        captured["timeout"] = timeout
        return 200, json.dumps(_assistant_payload()).encode("utf-8")

    client = LlmClient(_provider(), http_post=fake_post, timeout_seconds=30)
    result = client.chat_completions(
        [ChatMessage(role="user", content="ping")],
        temperature=0.0,
    )

    assert result.message.content == "hello"
    assert result.finish_reason == "stop"
    assert result.model == "llama3.1"
    assert captured["url"].endswith("/v1/chat/completions")
    assert "Authorization" not in captured["headers"]
    assert captured["body"]["model"] == "llama3.1"
    assert captured["body"]["messages"][0]["content"] == "ping"
    assert captured["body"]["temperature"] == 0.0
    assert captured["timeout"] == 30.0


def test_openrouter_sends_bearer_and_attribution_headers() -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        captured["headers"] = headers
        return 200, json.dumps(_assistant_payload(content="ok")).encode("utf-8")

    client = LlmClient(
        _provider(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4",
            api_key_env="OPENROUTER_API_KEY",
        ),
        environ={"OPENROUTER_API_KEY": "sk-test-secret"},
        http_post=fake_post,
    )
    client.chat_completions([{"role": "user", "content": "hi"}])

    assert captured["headers"]["Authorization"] == "Bearer sk-test-secret"
    assert captured["headers"]["X-Title"] == "Parcae"
    assert "HTTP-Referer" in captured["headers"]


def test_missing_api_key_env_raises() -> None:
    client = LlmClient(
        _provider(api_key_env="OPENROUTER_API_KEY"),
        environ={},
        http_post=lambda *a, **k: (200, b"{}"),
    )
    with pytest.raises(LlmError, match="OPENROUTER_API_KEY"):
        client.chat_completions([ChatMessage(role="user", content="x")])


def test_parses_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "catalog",
                "arguments": '{"kind":"transforms"}',
            },
        }
    ]

    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        req = json.loads(body.decode("utf-8"))
        assert req["tools"][0]["type"] == "function"
        assert req["tool_choice"] == "auto"
        return 200, json.dumps(
            _assistant_payload(content=None, tool_calls=tool_calls, finish_reason="tool_calls")
        ).encode("utf-8")

    client = LlmClient(_provider(), http_post=fake_post)
    result = client.chat_completions(
        [ChatMessage(role="system", content="use tools")],
        tools=[{"type": "function", "function": {"name": "catalog", "parameters": {}}}],
        tool_choice="auto",
    )
    assert result.has_tool_calls
    assert result.finish_reason == "tool_calls"
    assert result.message.tool_calls == (
        ToolCall(id="call_1", name="catalog", arguments='{"kind":"transforms"}'),
    )


def test_roundtrip_assistant_tool_message_serialization() -> None:
    msg = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=(ToolCall(id="c1", name="score", arguments='{"score_id":"ic_mod29"}'),),
    )
    wire = msg.to_openai_dict()
    assert wire["tool_calls"][0]["function"]["name"] == "score"

    tool_result = ChatMessage(
        role="tool",
        tool_call_id="c1",
        content='{"ok":true}',
    )
    assert tool_result.to_openai_dict()["tool_call_id"] == "c1"


def test_http_error_surfaces_status() -> None:
    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        return 401, b'{"error":{"message":"bad key"}}'

    client = LlmClient(_provider(), http_post=fake_post)
    with pytest.raises(LlmError, match="HTTP 401"):
        client.chat_completions([ChatMessage(role="user", content="x")])


def test_malformed_choices_raise() -> None:
    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        return 200, b'{"choices":[]}'

    client = LlmClient(_provider(), http_post=fake_post)
    with pytest.raises(LlmError, match="choices"):
        client.chat_completions([ChatMessage(role="user", content="x")])


def test_from_agent_config_factory() -> None:
    from pathlib import Path

    cfg = agent_config_from_mapping(
        {
            "schema": "parcae.agent_config.v0",
            "provider": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_env": None,
                "model": "m",
            },
            "parcae_bin_dir": str(Path.cwd()),
            "data_dir": str(Path.cwd()),
            "workspace": "ws-a",
            "budgets": {
                "max_steps": 1,
                "max_tool_calls": 1,
                "max_wall_seconds": 1,
            },
        }
    )

    seen: list[str] = []

    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        seen.append(json.loads(body.decode("utf-8"))["model"])
        return 200, json.dumps(_assistant_payload(content="z")).encode("utf-8")

    client = LlmClient.from_agent_config(cfg, http_post=fake_post)
    assert client.provider.model == "m"
    assert client.chat_completions([ChatMessage(role="user", content="q")]).message.content == "z"
    assert seen == ["m"]


def test_extra_headers_override_defaults() -> None:
    captured: dict[str, str] = {}

    def fake_post(
        url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        captured.update(headers)
        return 200, json.dumps(_assistant_payload()).encode("utf-8")

    client = LlmClient(
        _provider(base_url="https://openrouter.ai/api/v1", api_key_env="K"),
        environ={"K": "tok"},
        extra_headers={"X-Title": "Custom"},
        http_post=fake_post,
    )
    client.chat_completions([ChatMessage(role="user", content="x")])
    assert captured["X-Title"] == "Custom"
