"""CLI command implementations for `parcae-agent`."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from parcae_agent.allowlist import TOOL_BINARY
from parcae_agent.config import AgentConfig, AgentConfigError, load_agent_config
from parcae_agent.llm import ChatMessage, LlmClient, LlmError
from parcae_agent.loop import AgentLoop, AgentRunResult, AgentStep
from parcae_agent.tool_bridge import ToolBridge

LlmFactory = Callable[[AgentConfig], LlmClient]
BridgeFactory = Callable[[AgentConfig], ToolBridge]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    critical: bool = True


def load_config_or_exit(path: Path, *, stderr: TextIO = sys.stderr) -> AgentConfig | None:
    try:
        return load_agent_config(path)
    except AgentConfigError as exc:
        print(f"error: {exc}", file=stderr)
        return None


def resolve_prompt(
    *,
    prompt: str | None,
    prompt_file: Path | None,
    stdin: TextIO = sys.stdin,
) -> str:
    sources = sum(1 for x in (prompt, prompt_file) if x is not None)
    if sources > 1:
        raise ValueError("use only one of --prompt or --prompt-file")
    if prompt is not None:
        text = prompt
    elif prompt_file is not None:
        text = prompt_file.read_text(encoding="utf-8")
    else:
        if stdin.isatty():
            raise ValueError("missing prompt: pass --prompt, --prompt-file, or pipe stdin")
        text = stdin.read()
    text = text.strip()
    if not text:
        raise ValueError("prompt is empty")
    return text


def cmd_run(
    config_path: Path,
    *,
    prompt: str | None = None,
    prompt_file: Path | None = None,
    as_json: bool = False,
    verbose: bool = False,
    temperature: float = 0.0,
    llm_factory: LlmFactory | None = None,
    bridge_factory: BridgeFactory | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    stdin: TextIO = sys.stdin,
) -> int:
    cfg = load_config_or_exit(config_path, stderr=stderr)
    if cfg is None:
        return 2

    try:
        user_prompt = resolve_prompt(prompt=prompt, prompt_file=prompt_file, stdin=stdin)
    except ValueError as exc:
        print(f"error: {exc}", file=stderr)
        return 2

    llm = (llm_factory or _default_llm)(cfg)
    bridge = (bridge_factory or _default_bridge)(cfg)

    def on_step(step: AgentStep) -> None:
        if not verbose:
            return
        print(f"[step {step.index}] assistant tool_calls={len(step.assistant.tool_calls)}", file=stderr)
        for inv in step.tool_results:
            status = "ok" if inv.ok else "fail"
            print(f"  tool {inv.tool} → {status} (exit {inv.returncode})", file=stderr)

    loop = AgentLoop(cfg, llm, bridge, temperature=temperature, on_step=on_step)
    result = loop.run(user_prompt)
    _print_run_result(result, as_json=as_json, stdout=stdout, stderr=stderr)
    return result.exit_code


def cmd_providers_test(
    config_path: Path,
    *,
    as_json: bool = False,
    llm_factory: LlmFactory | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    cfg = load_config_or_exit(config_path, stderr=stderr)
    if cfg is None:
        return 2

    llm = (llm_factory or _default_llm)(cfg)
    try:
        completion = llm.chat_completions(
            [
                ChatMessage(
                    role="user",
                    content="Reply with the single word pong and nothing else.",
                )
            ],
            temperature=0.0,
            max_tokens=16,
        )
    except LlmError as exc:
        payload = {
            "ok": False,
            "base_url": cfg.provider.base_url,
            "model": cfg.provider.model,
            "error": str(exc),
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False), file=stdout)
        else:
            print(f"fail: provider test ({cfg.provider.base_url})", file=stderr)
            print(f"  error: {exc}", file=stderr)
        return 2

    content = (completion.message.content or "").strip()
    payload = {
        "ok": True,
        "base_url": cfg.provider.base_url,
        "model": completion.model or cfg.provider.model,
        "reply": content,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False), file=stdout)
    else:
        print(f"ok: provider {cfg.provider.base_url}", file=stdout)
        print(f"  model = {payload['model']}", file=stdout)
        print(f"  reply = {content!r}", file=stdout)
    return 0


def cmd_doctor(
    config_path: Path,
    *,
    as_json: bool = False,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    cfg = load_config_or_exit(config_path, stderr=stderr)
    if cfg is None:
        return 2

    env = environ if environ is not None else os.environ
    checks = run_doctor_checks(cfg, environ=env)

    if as_json:
        print(
            json.dumps(
                {
                    "ok": all(c.ok for c in checks if c.critical),
                    "checks": [
                        {
                            "name": c.name,
                            "ok": c.ok,
                            "detail": c.detail,
                            "critical": c.critical,
                        }
                        for c in checks
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=stdout,
        )
    else:
        for check in checks:
            mark = "ok" if check.ok else ("WARN" if not check.critical else "FAIL")
            print(f"{mark:4}  {check.name}: {check.detail}", file=stdout)

    critical_fail = any(not c.ok and c.critical for c in checks)
    warn_only = any(not c.ok and not c.critical for c in checks)
    if critical_fail:
        return 2
    if warn_only:
        return 1
    return 0


def run_doctor_checks(
    cfg: AgentConfig,
    *,
    environ: Mapping[str, str],
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []

    checks.append(
        DoctorCheck(
            "config",
            True,
            f"schema={cfg.schema} workspace={cfg.workspace}",
        )
    )

    data_ok = cfg.data_dir.is_dir()
    checks.append(
        DoctorCheck(
            "data_dir",
            data_ok,
            str(cfg.data_dir) if data_ok else f"missing directory: {cfg.data_dir}",
        )
    )
    fixtures = cfg.data_dir / "fixtures"
    checks.append(
        DoctorCheck(
            "fixtures",
            fixtures.is_dir(),
            str(fixtures) if fixtures.is_dir() else f"missing: {fixtures}",
            critical=False,
        )
    )

    bin_ok = cfg.parcae_bin_dir.is_dir()
    checks.append(
        DoctorCheck(
            "parcae_bin_dir",
            bin_ok,
            str(cfg.parcae_bin_dir)
            if bin_ok
            else f"missing directory: {cfg.parcae_bin_dir}",
        )
    )

    bridge = ToolBridge(cfg)
    missing: list[str] = []
    present: list[str] = []
    for name in sorted(set(TOOL_BINARY.values())):
        path = bridge.resolve_binary(name)
        if path.is_file():
            present.append(name)
        else:
            missing.append(name)
    checks.append(
        DoctorCheck(
            "binaries",
            not missing,
            (
                f"{len(present)} present"
                if not missing
                else f"missing: {', '.join(missing)}"
            ),
        )
    )

    if cfg.provider.api_key_env is None:
        checks.append(
            DoctorCheck(
                "api_key_env",
                True,
                "null (no Authorization header)",
                critical=False,
            )
        )
    else:
        key_name = cfg.provider.api_key_env
        set_ok = bool(environ.get(key_name))
        checks.append(
            DoctorCheck(
                "api_key_env",
                set_ok,
                f"{key_name} is set" if set_ok else f"{key_name} is unset or empty",
            )
        )

    checks.append(
        DoctorCheck(
            "provider.base_url",
            True,
            cfg.provider.base_url,
            critical=False,
        )
    )
    checks.append(
        DoctorCheck(
            "allow_cuda",
            True,
            str(cfg.allow_cuda).lower(),
            critical=False,
        )
    )
    return checks


def _default_llm(cfg: AgentConfig) -> LlmClient:
    return LlmClient.from_agent_config(cfg)


def _default_bridge(cfg: AgentConfig) -> ToolBridge:
    return ToolBridge(cfg, timeout_seconds=float(cfg.budgets.max_wall_seconds))


def _print_run_result(
    result: AgentRunResult,
    *,
    as_json: bool,
    stdout: TextIO,
    stderr: TextIO,
) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "stop_reason": result.stop_reason.value,
                    "exit_code": result.exit_code,
                    "tool_calls": result.tool_calls,
                    "steps": len(result.steps),
                    "wall_seconds": result.wall_seconds,
                    "final_text": result.final_text,
                    "error": result.error,
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=stdout,
        )
        return

    if result.final_text:
        print(result.final_text, file=stdout)
    if result.error:
        print(f"error: {result.error}", file=stderr)
    print(
        f"[{result.stop_reason.value}] tools={result.tool_calls} "
        f"steps={len(result.steps)} wall={result.wall_seconds:.2f}s",
        file=stderr,
    )
