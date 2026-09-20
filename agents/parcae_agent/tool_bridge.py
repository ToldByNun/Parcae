"""ToolBridge: allow-listed argv → subprocess → parcae.tool_response.v0.

Never runs free-form shell from model output (`shell=False`, argv built only
from the allow-list schema).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from parcae_agent.allowlist import (
    ALLOWED_TOOLS,
    BOOL_FLAGS,
    BRIDGE_OWNED_KEYS,
    DENIED_BINARIES,
    HYPOTHESIS_SUBCOMMAND,
    TOOL_ARG_KEYS,
    TOOL_BINARY,
    VALUE_FLAGS,
)
from parcae_agent.config import AgentConfig
from parcae_agent.llm import ToolCall

TOOL_RESPONSE_SCHEMA = "parcae.tool_response.v0"

SubprocessRunner = Callable[
    [list[str], float | None],
    "SubprocessResult",
]


class ToolBridgeError(ValueError):
    """Invalid tool name / arguments before a process is started."""


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ToolInvocationResult:
    tool: str
    argv: tuple[str, ...]
    returncode: int
    envelope: dict[str, Any]
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return bool(self.envelope.get("ok"))


def synthesize_envelope(
    tool: str,
    *,
    code: str,
    message: str,
    backend: str | None = None,
) -> dict[str, Any]:
    """Build a failure envelope when the CLI produced no usable JSON."""
    return {
        "schema": TOOL_RESPONSE_SCHEMA,
        "ok": False,
        "tool": tool,
        "backend": backend,
        "result": None,
        "error": {"code": code, "message": message},
    }


def parse_tool_response(stdout: str, *, tool: str) -> dict[str, Any]:
    """Parse stdout as `parcae.tool_response.v0`, else synthesize internal."""
    text = stdout.strip()
    if not text:
        return synthesize_envelope(
            tool,
            code="internal",
            message="CLI produced empty stdout (expected parcae.tool_response.v0)",
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return synthesize_envelope(
            tool,
            code="internal",
            message=f"CLI stdout is not JSON: {exc}",
        )
    if not isinstance(data, dict):
        return synthesize_envelope(
            tool,
            code="internal",
            message="CLI stdout JSON root must be an object",
        )
    schema = data.get("schema")
    if schema != TOOL_RESPONSE_SCHEMA:
        return synthesize_envelope(
            tool,
            code="internal",
            message=f"unexpected schema {schema!r} (want {TOOL_RESPONSE_SCHEMA})",
        )
    if "ok" not in data or "tool" not in data:
        return synthesize_envelope(
            tool,
            code="internal",
            message="CLI envelope missing required ok/tool fields",
        )
    return data


def _default_runner(argv: list[str], timeout: float | None) -> SubprocessResult:
    completed = subprocess.run(  # noqa: S603 — argv is allow-list built, not shell
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=timeout,
        check=False,
    )
    return SubprocessResult(
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


class ToolBridge:
    """Map allow-listed tool calls to Parcae CLI subprocesses."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        timeout_seconds: float | None = None,
        runner: SubprocessRunner | None = None,
    ) -> None:
        self._config = config
        self._timeout = timeout_seconds
        self._runner = runner or _default_runner

    @property
    def config(self) -> AgentConfig:
        return self._config

    def allowed_tools(self) -> frozenset[str]:
        return ALLOWED_TOOLS

    def build_argv(self, tool: str, arguments: Mapping[str, Any] | None = None) -> list[str]:
        """Validate arguments and return argv (executable path first)."""
        if tool not in ALLOWED_TOOLS:
            raise ToolBridgeError(f"tool not on allow-list: {tool}")

        args = dict(arguments or {})
        owned = BRIDGE_OWNED_KEYS & set(args.keys())
        if owned:
            raise ToolBridgeError(
                "arguments must not include bridge-owned keys: "
                + ", ".join(sorted(owned))
            )

        allowed_keys = TOOL_ARG_KEYS[tool]
        unknown = set(args.keys()) - allowed_keys
        if unknown:
            raise ToolBridgeError(
                f"unknown argument(s) for {tool}: " + ", ".join(sorted(unknown))
            )

        binary_name = TOOL_BINARY[tool]
        self._assert_binary_allowed(binary_name)
        exe = self.resolve_binary(binary_name)

        argv: list[str] = [str(exe)]
        if tool in HYPOTHESIS_SUBCOMMAND:
            argv.append(HYPOTHESIS_SUBCOMMAND[tool])

        # Bridge-owned injections (never from the model).
        argv.extend(["--data-dir", str(self._config.data_dir)])
        argv.append("--json")
        if tool.startswith("hypothesis_"):
            argv.extend(["--workspace", self._config.workspace])

        # CUDA opt-in only from agent config, and only when backend requests cuda.
        backend = args.get("backend")
        if backend is not None:
            if not isinstance(backend, str) or backend not in {"cpu", "cuda"}:
                raise ToolBridgeError("backend must be 'cpu' or 'cuda'")
            if backend == "cuda":
                if not self._config.allow_cuda:
                    raise ToolBridgeError(
                        "backend cuda requires allow_cuda: true in agent config"
                    )
                argv.append("--allow-cuda")

        # tokenize: positional input (file or -); optional --strict/--no-strict
        if tool == "tokenize":
            strict = args.pop("strict", True)
            if not isinstance(strict, bool):
                raise ToolBridgeError("strict must be a boolean")
            argv.append("--strict" if strict else "--no-strict")
            input_path = args.pop("input", None)
            if input_path is None:
                raise ToolBridgeError("tokenize requires input")
            argv.append(_require_str(input_path, "input"))
            if args:
                raise ToolBridgeError(
                    "internal: leftover tokenize args: " + ", ".join(sorted(args))
                )
            return argv

        for key, value in args.items():
            if key in BOOL_FLAGS:
                if not isinstance(value, bool):
                    raise ToolBridgeError(f"{key} must be a boolean")
                if value:
                    argv.append(BOOL_FLAGS[key])
                continue

            if key == "params_json" or key.endswith("_json") and key in VALUE_FLAGS:
                argv.extend([VALUE_FLAGS[key], _json_value(value, key)])
                continue

            if key not in VALUE_FLAGS:
                raise ToolBridgeError(f"internal: unmapped key {key}")
            argv.extend([VALUE_FLAGS[key], _stringify_value(value, key)])

        return argv

    def invoke(
        self,
        tool: str,
        arguments: Mapping[str, Any] | str | None = None,
    ) -> ToolInvocationResult:
        """Build argv, run subprocess, parse/synthesize tool_response envelope."""
        try:
            args_map = _coerce_arguments(arguments)
            argv = self.build_argv(tool, args_map)
        except ToolBridgeError as exc:
            envelope = synthesize_envelope(tool, code="policy", message=str(exc))
            return ToolInvocationResult(
                tool=tool,
                argv=(),
                returncode=2,
                envelope=envelope,
                stdout=json.dumps(envelope, ensure_ascii=False),
                stderr="",
            )

        try:
            completed = self._runner(argv, self._timeout)
        except subprocess.TimeoutExpired as exc:
            envelope = synthesize_envelope(
                tool,
                code="internal",
                message=f"CLI timed out after {self._timeout}s",
            )
            return ToolInvocationResult(
                tool=tool,
                argv=tuple(argv),
                returncode=2,
                envelope=envelope,
                stdout=exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, (bytes, bytearray))
                else (exc.stdout or ""),
                stderr=exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, (bytes, bytearray))
                else (exc.stderr or ""),
            )
        except OSError as exc:
            envelope = synthesize_envelope(
                tool,
                code="internal",
                message=f"failed to start CLI: {exc}",
            )
            return ToolInvocationResult(
                tool=tool,
                argv=tuple(argv),
                returncode=2,
                envelope=envelope,
                stdout="",
                stderr=str(exc),
            )

        envelope = parse_tool_response(completed.stdout, tool=tool)
        return ToolInvocationResult(
            tool=tool,
            argv=tuple(argv),
            returncode=completed.returncode,
            envelope=envelope,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def invoke_tool_call(self, tool_call: ToolCall) -> ToolInvocationResult:
        """Invoke from an LLM ToolCall (name + JSON arguments string)."""
        return self.invoke(tool_call.name, tool_call.arguments)

    def resolve_binary(self, binary_name: str) -> Path:
        self._assert_binary_allowed(binary_name)
        bin_dir = self._config.parcae_bin_dir
        candidates = [bin_dir / binary_name]
        if sys.platform == "win32":
            candidates.insert(0, bin_dir / f"{binary_name}.exe")
        else:
            candidates.append(bin_dir / f"{binary_name}.exe")
        for path in candidates:
            if path.is_file():
                return path.resolve()
        # Prefer platform-native name for the error path.
        preferred = candidates[0]
        return preferred.resolve()

    def _assert_binary_allowed(self, binary_name: str) -> None:
        stem = Path(binary_name).name
        if stem in DENIED_BINARIES:
            raise ToolBridgeError(f"binary is deny-listed: {stem}")


def _coerce_arguments(arguments: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        text = arguments.strip() or "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolBridgeError(f"tool arguments are not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ToolBridgeError("tool arguments JSON must be an object")
        return parsed
    raise ToolBridgeError(f"unsupported arguments type: {type(arguments)!r}")


def _require_str(value: Any, key: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ToolBridgeError(f"{key} must be a non-empty string")
    if "\x00" in value:
        raise ToolBridgeError(f"{key} must not contain NUL")
    return value


def _stringify_value(value: Any, key: str) -> str:
    if isinstance(value, bool):
        raise ToolBridgeError(f"{key} must not be a boolean (use a flag key)")
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return _require_str(value, key)
    raise ToolBridgeError(f"{key} must be a string or number")


def _json_value(value: Any, key: str) -> str:
    if isinstance(value, str):
        # Already a JSON text blob from the model.
        text = value.strip()
        if not text:
            raise ToolBridgeError(f"{key} must be non-empty JSON")
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolBridgeError(f"{key} is not valid JSON: {exc}") from exc
        return text
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ToolBridgeError(f"{key} is not JSON-serializable: {exc}") from exc
