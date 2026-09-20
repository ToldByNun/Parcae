"""Workspace transcript persistence (`parcae.transcript_step.v0`).

Hypotheses are persisted only via allow-listed `hypothesis_*` tools (ToolBridge).
This module records agent-run JSONL under `data/workspaces/<id>/transcripts/`.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from parcae_agent.config import AgentConfig
from parcae_agent.tool_bridge import ToolInvocationResult

TRANSCRIPT_SCHEMA = "parcae.transcript_step.v0"
_SUMMARY_MAX = 240
_SECRETISH = re.compile(r"(?i)(api[_-]?key|authorization|bearer\s+\S+|sk-[A-Za-z0-9]{8,})")


class TranscriptError(RuntimeError):
    """Failed to create or append a transcript under the workspace."""


@dataclass(frozen=True, slots=True)
class TranscriptStep:
    schema: str
    workspace_id: str
    run_id: str
    seq: int
    utc: str
    role: str
    tool: str | None
    ok: bool | None
    summary: str
    hypothesis_id: str | None
    envelope_ref: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "workspace_id": self.workspace_id,
            "run_id": self.run_id,
            "seq": self.seq,
            "utc": self.utc,
            "role": self.role,
            "tool": self.tool,
            "ok": self.ok,
            "summary": self.summary,
            "hypothesis_id": self.hypothesis_id,
            "envelope_ref": self.envelope_ref,
        }


class TranscriptWriter:
    """Append-only JSONL transcript for one agent run."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        run_id: str | None = None,
        save_envelopes: bool = True,
        utc_now: Callable[[], str] | None = None,
    ) -> None:
        self._config = config
        self._run_id = run_id or secrets.token_hex(4)
        self._save_envelopes = save_envelopes
        self._utc_now = utc_now or _utc_now
        self._seq = 0
        self._path = self._create_file()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def workspace_root(self) -> Path:
        return self._config.data_dir / "workspaces" / self._config.workspace

    @property
    def transcripts_dir(self) -> Path:
        return self.workspace_root / "transcripts"

    def append(
        self,
        role: str,
        *,
        summary: str,
        tool: str | None = None,
        ok: bool | None = None,
        hypothesis_id: str | None = None,
        envelope_ref: str | None = None,
    ) -> TranscriptStep:
        if role not in {"system", "user", "assistant", "tool", "finish"}:
            raise TranscriptError(f"invalid transcript role: {role}")
        step = TranscriptStep(
            schema=TRANSCRIPT_SCHEMA,
            workspace_id=self._config.workspace,
            run_id=self._run_id,
            seq=self._seq,
            utc=self._utc_now(),
            role=role,
            tool=tool,
            ok=ok,
            summary=redact_summary(summary),
            hypothesis_id=hypothesis_id,
            envelope_ref=envelope_ref,
        )
        line = json.dumps(step.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        self._seq += 1
        return step

    def append_tool(
        self,
        invocation: ToolInvocationResult,
        *,
        arguments: str | Mapping[str, Any] | None = None,
    ) -> TranscriptStep:
        hyp_id = extract_hypothesis_id(invocation.tool, arguments, invocation.envelope)
        envelope_ref = None
        if self._save_envelopes:
            envelope_ref = self._save_envelope(invocation)
        return self.append(
            "tool",
            summary=_tool_summary(invocation),
            tool=invocation.tool,
            ok=invocation.ok,
            hypothesis_id=hyp_id,
            envelope_ref=envelope_ref,
        )

    def _create_file(self) -> Path:
        root = self.workspace_root.resolve()
        data_root = self._config.data_dir.resolve()
        try:
            root.relative_to(data_root / "workspaces")
        except ValueError as exc:
            raise TranscriptError(
                f"workspace path escapes data_dir/workspaces: {root}"
            ) from exc
        if "fixtures" in root.parts:
            raise TranscriptError("refusing transcript write under fixtures/")

        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        stamp = _filename_stamp(self._utc_now())
        path = self.transcripts_dir / f"{stamp}_{self._run_id}.jsonl"
        path.touch(exist_ok=False)
        return path

    def _save_envelope(self, invocation: ToolInvocationResult) -> str:
        env_dir = self.transcripts_dir / "envelopes" / self._run_id
        env_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self._seq:04d}_{invocation.tool}.json"
        path = env_dir / name
        path.write_text(
            json.dumps(invocation.envelope, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rel = Path("transcripts") / "envelopes" / self._run_id / name
        return rel.as_posix()


def redact_summary(text: str, *, max_len: int = _SUMMARY_MAX) -> str:
    cleaned = _SECRETISH.sub("[redacted]", text.replace("\n", " ").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def extract_hypothesis_id(
    tool: str,
    arguments: str | Mapping[str, Any] | None,
    envelope: Mapping[str, Any] | None = None,
) -> str | None:
    if not tool.startswith("hypothesis_"):
        return None
    args = _coerce_args(arguments)
    hid = args.get("id")
    if isinstance(hid, str) and hid.strip():
        return hid.strip()
    if envelope and isinstance(envelope.get("result"), Mapping):
        rid = envelope["result"].get("id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    return None


def load_transcript(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL transcript file into a list of step dicts."""
    steps: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranscriptError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict) or obj.get("schema") != TRANSCRIPT_SCHEMA:
            raise TranscriptError(f"{path}:{line_no}: expected {TRANSCRIPT_SCHEMA}")
        steps.append(obj)
    return steps


def _tool_summary(invocation: ToolInvocationResult) -> str:
    if invocation.ok:
        return f"{invocation.tool} ok"
    err = invocation.envelope.get("error") if isinstance(invocation.envelope, dict) else None
    if isinstance(err, Mapping):
        code = err.get("code", "error")
        msg = err.get("message", "")
        return redact_summary(f"{invocation.tool} {code}: {msg}")
    return f"{invocation.tool} failed"


def _coerce_args(arguments: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    try:
        parsed = json.loads(arguments.strip() or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _filename_stamp(utc: str) -> str:
    # "2026-09-20T01:02:03Z" → "20260920T010203Z"
    if utc.endswith("Z") and "T" in utc:
        return utc.replace("-", "").replace(":", "")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
