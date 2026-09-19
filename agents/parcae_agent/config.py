"""Load and validate `parcae.agent_config.v0`."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from parcae_agent import SCHEMA_ID

_WORKSPACE_ID_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")


class AgentConfigError(ValueError):
    """Invalid or incomplete agent config."""


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    base_url: str
    model: str
    api_key_env: str | None = None

    def resolve_api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        """Read the named env var; never stores the secret on this object."""
        if self.api_key_env is None:
            return None
        env = environ if environ is not None else os.environ
        value = env.get(self.api_key_env)
        if value is None or value == "":
            raise AgentConfigError(
                f"environment variable {self.api_key_env!r} is unset or empty"
            )
        return value


@dataclass(frozen=True, slots=True)
class Budgets:
    max_steps: int
    max_tool_calls: int
    max_wall_seconds: int


@dataclass(frozen=True, slots=True)
class AgentConfig:
    provider: ProviderConfig
    parcae_bin_dir: Path
    data_dir: Path
    workspace: str
    budgets: Budgets
    allow_cuda: bool = False
    schema: str = SCHEMA_ID
    source_path: Path | None = None

    def to_public_dict(self) -> dict[str, Any]:
        """JSON/YAML-safe dump without resolving secrets."""
        return {
            "schema": self.schema,
            "provider": {
                "base_url": self.provider.base_url,
                "api_key_env": self.provider.api_key_env,
                "model": self.provider.model,
            },
            "parcae_bin_dir": str(self.parcae_bin_dir),
            "data_dir": str(self.data_dir),
            "workspace": self.workspace,
            "allow_cuda": self.allow_cuda,
            "budgets": asdict(self.budgets),
        }


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load YAML or JSON from `path` and validate as `parcae.agent_config.v0`."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise AgentConfigError(f"config file not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    try:
        if suffix == ".json":
            raw: Any = json.loads(raw_text)
        else:
            raw = yaml.safe_load(raw_text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise AgentConfigError(f"failed to parse config: {exc}") from exc

    if not isinstance(raw, dict):
        raise AgentConfigError("config root must be a mapping/object")

    return agent_config_from_mapping(raw, source_path=config_path)


def agent_config_from_mapping(
    raw: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    base_dir: Path | None = None,
) -> AgentConfig:
    """Validate a mapping (already-parsed YAML/JSON). Relative paths resolve
    against `base_dir` (default: parent of `source_path`, else cwd)."""
    schema = raw.get("schema")
    if schema != SCHEMA_ID:
        raise AgentConfigError(
            f"schema must be {SCHEMA_ID!r}, got {schema!r}"
        )

    provider_raw = raw.get("provider")
    if not isinstance(provider_raw, Mapping):
        raise AgentConfigError("provider must be a mapping")

    base_url = _require_nonempty_str(provider_raw, "base_url", "provider.base_url")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        raise AgentConfigError("provider.base_url must start with http:// or https://")

    model = _require_nonempty_str(provider_raw, "model", "provider.model")

    api_key_env = provider_raw.get("api_key_env", None)
    if api_key_env is not None:
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise AgentConfigError(
                "provider.api_key_env must be null or a non-empty env var name"
            )
        api_key_env = api_key_env.strip()

    provider = ProviderConfig(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key_env=api_key_env,
    )

    resolve_base = base_dir
    if resolve_base is None and source_path is not None:
        resolve_base = source_path.parent
    if resolve_base is None:
        resolve_base = Path.cwd()

    bin_dir = _resolve_dir_field(raw, "parcae_bin_dir", resolve_base)
    data_dir = _resolve_dir_field(raw, "data_dir", resolve_base)
    _deny_data_dir_under_fixtures(data_dir)

    workspace = _require_nonempty_str(raw, "workspace", "workspace")
    if not _WORKSPACE_ID_RE.fullmatch(workspace):
        raise AgentConfigError(
            "workspace must match ^[a-z_][a-z0-9_-]{0,63}$ "
            f"(got {workspace!r})"
        )

    allow_cuda = raw.get("allow_cuda", False)
    if not isinstance(allow_cuda, bool):
        raise AgentConfigError("allow_cuda must be a boolean")

    budgets_raw = raw.get("budgets")
    if not isinstance(budgets_raw, Mapping):
        raise AgentConfigError("budgets must be a mapping")
    budgets = Budgets(
        max_steps=_require_positive_int(budgets_raw, "max_steps", "budgets.max_steps"),
        max_tool_calls=_require_positive_int(
            budgets_raw, "max_tool_calls", "budgets.max_tool_calls"
        ),
        max_wall_seconds=_require_positive_int(
            budgets_raw, "max_wall_seconds", "budgets.max_wall_seconds"
        ),
    )

    unknown = set(raw.keys()) - {
        "schema",
        "provider",
        "parcae_bin_dir",
        "data_dir",
        "workspace",
        "allow_cuda",
        "budgets",
    }
    if unknown:
        raise AgentConfigError(
            "unknown top-level field(s): " + ", ".join(sorted(unknown))
        )

    return AgentConfig(
        schema=SCHEMA_ID,
        provider=provider,
        parcae_bin_dir=bin_dir,
        data_dir=data_dir,
        workspace=workspace,
        allow_cuda=allow_cuda,
        budgets=budgets,
        source_path=source_path,
    )


def _require_nonempty_str(raw: Mapping[str, Any], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _require_positive_int(raw: Mapping[str, Any], key: str, label: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentConfigError(f"{label} must be an integer >= 1")
    if value < 1:
        raise AgentConfigError(f"{label} must be an integer >= 1")
    return value


def _resolve_dir_field(
    raw: Mapping[str, Any], key: str, base_dir: Path
) -> Path:
    text = _require_nonempty_str(raw, key, key)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _deny_data_dir_under_fixtures(data_dir: Path) -> None:
    """Mirror AgentPolicy: data_dir must not sit under a `fixtures/` tree."""
    cur = data_dir
    while True:
        if cur.name == "fixtures":
            raise AgentConfigError(
                "data_dir must not be under fixtures/ (refusing agent config)"
            )
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
