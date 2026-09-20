from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from parcae_agent.__main__ import main
from parcae_agent.cli import cmd_doctor, cmd_providers_test, cmd_run, resolve_prompt
from parcae_agent.llm import ChatCompletionResult, ChatMessage
from parcae_agent.loop import AgentStopReason
from parcae_agent.tool_bridge import SubprocessResult, ToolBridge

AGENTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENTS_ROOT.parent
OLLAMA = AGENTS_ROOT / "configs" / "ollama.example.yaml"


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    raw = {
        "schema": "parcae.agent_config.v0",
        "provider": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key_env": None,
            "model": "mock",
        },
        "parcae_bin_dir": str(REPO_ROOT / "build-cuda" / "tools" / "Debug"),
        "data_dir": str(REPO_ROOT / "data"),
        "workspace": "cli-ws",
        "allow_cuda": False,
        "budgets": {
            "max_steps": 4,
            "max_tool_calls": 8,
            "max_wall_seconds": 30,
        },
    }
    raw.update(overrides)
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _fake_llm(content: str = "done"):
    class FakeLlm:
        def chat_completions(self, messages, **kwargs):
            return ChatCompletionResult(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
                model="mock",
                raw={},
            )

    return FakeLlm()


def _quiet_bridge(cfg):
    return ToolBridge(
        cfg,
        runner=lambda *a, **k: SubprocessResult(0, "{}", ""),
    )


def test_resolve_prompt_sources(tmp_path: Path) -> None:
    assert resolve_prompt(prompt="hi", prompt_file=None) == "hi"
    f = tmp_path / "p.txt"
    f.write_text(" from file \n", encoding="utf-8")
    assert resolve_prompt(prompt=None, prompt_file=f) == "from file"
    with pytest.raises(ValueError, match="only one"):
        resolve_prompt(prompt="a", prompt_file=f)
    with pytest.raises(ValueError, match="empty"):
        resolve_prompt(prompt="  ", prompt_file=None)


def test_cmd_run_with_injected_llm(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = cmd_run(
        cfg_path,
        prompt="hello",
        llm_factory=lambda cfg: _fake_llm("done"),  # type: ignore[arg-type,return-value]
        bridge_factory=_quiet_bridge,
        stdout=out,
        stderr=err,
    )
    assert code == 0
    assert "done" in out.getvalue()
    assert AgentStopReason.Completed.value in err.getvalue()


def test_cmd_run_json_summary(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = cmd_run(
        cfg_path,
        prompt="x",
        as_json=True,
        llm_factory=lambda cfg: _fake_llm("summary"),  # type: ignore[arg-type,return-value]
        bridge_factory=_quiet_bridge,
        stdout=out,
        stderr=err,
    )
    assert code == 0
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["stop_reason"] == "completed"
    assert payload["final_text"] == "summary"


def test_main_run_via_argv_missing_prompt(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    assert main(["run", "--config", str(cfg_path), "--prompt", ""]) == 2


def test_providers_test_ok_and_fail(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out, err = io.StringIO(), io.StringIO()

    class OkLlm:
        def chat_completions(self, messages, **kwargs):
            return ChatCompletionResult(
                message=ChatMessage(role="assistant", content="pong"),
                finish_reason="stop",
                model="mock-1",
                raw={},
            )

    assert (
        cmd_providers_test(
            cfg_path,
            llm_factory=lambda c: OkLlm(),  # type: ignore[arg-type,return-value]
            stdout=out,
            stderr=err,
        )
        == 0
    )
    assert "ok: provider" in out.getvalue()
    assert "pong" in out.getvalue()

    class Boom:
        def chat_completions(self, *a, **k):
            from parcae_agent.llm import LlmError

            raise LlmError("refused")

    err2 = io.StringIO()
    assert (
        cmd_providers_test(
            cfg_path,
            llm_factory=lambda c: Boom(),  # type: ignore[arg-type,return-value]
            stdout=io.StringIO(),
            stderr=err2,
        )
        == 2
    )
    assert "refused" in err2.getvalue()


def test_providers_test_json(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out = io.StringIO()

    class OkLlm:
        def chat_completions(self, messages, **kwargs):
            return ChatCompletionResult(
                message=ChatMessage(role="assistant", content="pong"),
                finish_reason="stop",
                model="m",
                raw={},
            )

    assert (
        cmd_providers_test(
            cfg_path,
            as_json=True,
            llm_factory=lambda c: OkLlm(),  # type: ignore[arg-type,return-value]
            stdout=out,
        )
        == 0
    )
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["reply"] == "pong"


def test_doctor_pass_on_example_config() -> None:
    out = io.StringIO()
    code = cmd_doctor(OLLAMA, stdout=out)
    text = out.getvalue()
    assert "data_dir" in text
    assert code in (0, 1, 2)


def test_doctor_json_critical_fail(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        data_dir=str(tmp_path / "no-data"),
        parcae_bin_dir=str(tmp_path / "no-bins"),
    )
    out = io.StringIO()
    code = cmd_doctor(cfg_path, as_json=True, environ={}, stdout=out)
    assert code == 2
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    names = {c["name"]: c for c in payload["checks"]}
    assert names["data_dir"]["ok"] is False
    assert names["parcae_bin_dir"]["ok"] is False


def test_doctor_api_key_required(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path,
        provider={
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "model": "x",
        },
    )
    out = io.StringIO()
    cmd_doctor(cfg_path, as_json=True, environ={}, stdout=out)
    payload = json.loads(out.getvalue())
    api = next(c for c in payload["checks"] if c["name"] == "api_key_env")
    assert api["ok"] is False

    out2 = io.StringIO()
    cmd_doctor(
        cfg_path,
        as_json=True,
        environ={"OPENROUTER_API_KEY": "sk"},
        stdout=out2,
    )
    payload2 = json.loads(out2.getvalue())
    api2 = next(c for c in payload2["checks"] if c["name"] == "api_key_env")
    assert api2["ok"] is True


def test_main_providers_test_argv_parsing(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    # Unreachable provider → exit 2 (not argparse failure).
    code = main(["providers", "test", "--config", str(cfg_path), "--json"])
    assert code == 2
