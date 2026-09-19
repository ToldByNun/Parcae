"""OpenAI-compatible chat completions client (local + OpenRouter).

One HTTP adapter: `POST {base_url}/chat/completions`. The same client works for
Ollama / LM Studio / llama.cpp and cloud gateways that speak the OpenAI schema.
Secrets come from `ProviderConfig.resolve_api_key` and are never stored on the
client beyond the request Authorization header.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urljoin

from parcae_agent.config import AgentConfig, AgentConfigError, ProviderConfig

HttpPost = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


class LlmError(RuntimeError):
    """Transport, HTTP, or response-shape failure talking to the LLM provider."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON object as a string (OpenAI wire format)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_openai_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.name is not None:
            out["name"] = self.name
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return out


@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    message: ChatMessage
    finish_reason: str | None
    model: str | None
    raw: dict[str, Any] = field(repr=False)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)


def _default_http_post(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read() if exc.fp is not None else b""
        return int(exc.code), payload
    except urllib.error.URLError as exc:
        raise LlmError(f"LLM request failed: {exc.reason}") from exc


class LlmClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        timeout_seconds: float = 120.0,
        extra_headers: Mapping[str, str] | None = None,
        environ: Mapping[str, str] | None = None,
        http_post: HttpPost | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self._provider = provider
        self._timeout = float(timeout_seconds)
        self._extra_headers = dict(extra_headers or {})
        self._environ = environ
        self._http_post = http_post or _default_http_post

    @classmethod
    def from_agent_config(
        cls,
        config: AgentConfig,
        *,
        timeout_seconds: float = 120.0,
        extra_headers: Mapping[str, str] | None = None,
        environ: Mapping[str, str] | None = None,
        http_post: HttpPost | None = None,
    ) -> LlmClient:
        return cls(
            config.provider,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            environ=environ,
            http_post=http_post,
        )

    @property
    def provider(self) -> ProviderConfig:
        return self._provider

    @property
    def chat_completions_url(self) -> str:
        base = self._provider.base_url.rstrip("/") + "/"
        return urljoin(base, "chat/completions")

    def chat_completions(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletionResult:
        body: dict[str, Any] = {
            "model": self._provider.model,
            "messages": [_coerce_message(m) for m in messages],
        }
        if tools is not None:
            body["tools"] = list(tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        headers = self._build_headers()
        raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        status, payload = self._http_post(
            self.chat_completions_url, headers, raw_body, self._timeout
        )
        return self._parse_response(status, payload)

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            api_key = self._provider.resolve_api_key(self._environ)
        except AgentConfigError as exc:
            raise LlmError(str(exc)) from exc
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        # OpenRouter (and some gateways) accept optional attribution headers.
        host = self._provider.base_url.lower()
        if "openrouter.ai" in host:
            headers.setdefault("X-Title", "Parcae")
            headers.setdefault("HTTP-Referer", "https://github.com/ToldByNun/Parcae")

        for key, value in self._extra_headers.items():
            headers[key] = value
        return headers

    def _parse_response(self, status: int, payload: bytes) -> ChatCompletionResult:
        text = payload.decode("utf-8", errors="replace")
        if status < 200 or status >= 300:
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            raise LlmError(f"LLM HTTP {status}: {snippet or '(empty body)'}")

        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise LlmError(f"LLM response is not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LlmError("LLM response root must be a JSON object")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmError("LLM response missing non-empty choices[]")
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise LlmError("LLM choices[0] must be an object")

        message_raw = choice0.get("message")
        if not isinstance(message_raw, dict):
            raise LlmError("LLM choices[0].message must be an object")

        finish_reason = choice0.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise LlmError("LLM finish_reason must be a string when present")

        model = data.get("model")
        if model is not None and not isinstance(model, str):
            model = None

        return ChatCompletionResult(
            message=_message_from_openai(message_raw),
            finish_reason=finish_reason,
            model=model,
            raw=data,
        )


def _coerce_message(message: ChatMessage | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return message.to_openai_dict()
    if isinstance(message, Mapping):
        return dict(message)
    raise TypeError(f"unsupported message type: {type(message)!r}")


def _message_from_openai(raw: Mapping[str, Any]) -> ChatMessage:
    role = raw.get("role")
    if not isinstance(role, str) or not role:
        raise LlmError("assistant message missing role")

    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        # Some providers return a list of content parts; flatten text if present.
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts) if parts else None
        else:
            raise LlmError("assistant message content must be a string or content parts")

    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        raise LlmError("assistant message name must be a string when present")

    tool_call_id = raw.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise LlmError("tool_call_id must be a string when present")

    tool_calls_raw = raw.get("tool_calls") or []
    if not isinstance(tool_calls_raw, list):
        raise LlmError("tool_calls must be a list when present")

    tool_calls: list[ToolCall] = []
    for item in tool_calls_raw:
        if not isinstance(item, Mapping):
            raise LlmError("each tool_call must be an object")
        tc_id = item.get("id")
        fn = item.get("function")
        if not isinstance(tc_id, str) or not tc_id:
            raise LlmError("tool_call.id must be a non-empty string")
        if not isinstance(fn, Mapping):
            raise LlmError("tool_call.function must be an object")
        fn_name = fn.get("name")
        arguments = fn.get("arguments", "{}")
        if not isinstance(fn_name, str) or not fn_name:
            raise LlmError("tool_call.function.name must be a non-empty string")
        if not isinstance(arguments, str):
            # Some servers return a JSON object; normalize to wire string.
            arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(ToolCall(id=tc_id, name=fn_name, arguments=arguments))

    return ChatMessage(
        role=role,
        content=content,
        name=name,
        tool_call_id=tool_call_id,
        tool_calls=tuple(tool_calls),
    )
