from __future__ import annotations

import json
from pathlib import Path

from parcae_agent.__main__ import main

AGENTS_ROOT = Path(__file__).resolve().parents[1]


def test_check_config_ok(capsys) -> None:
    path = AGENTS_ROOT / "configs" / "ollama.example.yaml"
    assert main(["check-config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "ok: parcae.agent_config.v0" in out
    assert "llama3.1" in out


def test_check_config_json(capsys) -> None:
    path = AGENTS_ROOT / "configs" / "ollama.example.yaml"
    assert main(["check-config", "--json", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "parcae.agent_config.v0"
    assert payload["provider"]["model"] == "llama3.1"


def test_check_config_invalid(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: nope\n", encoding="utf-8")
    assert main(["check-config", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "error:" in err
