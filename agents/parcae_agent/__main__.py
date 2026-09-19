"""Minimal CLI entry for scaffold commands (full `run` arrives later)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parcae_agent import SCHEMA_ID, __version__
from parcae_agent.config import AgentConfigError, load_agent_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="parcae-agent",
        description="Parcae CMD Liber Primus tool-use agent",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"parcae-agent {__version__} ({SCHEMA_ID})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check-config",
        help=f"Validate a {SCHEMA_ID} YAML/JSON file",
    )
    check.add_argument(
        "path",
        type=Path,
        help="Path to agent config file",
    )
    check.add_argument(
        "--json",
        action="store_true",
        help="Print the public (secret-free) config as JSON",
    )

    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _cmd_check_config(args.path, as_json=args.json)
    parser.error(f"unknown command: {args.command}")
    return 2


def _cmd_check_config(path: Path, *, as_json: bool) -> int:
    try:
        cfg = load_agent_config(path)
    except AgentConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if as_json:
        print(json.dumps(cfg.to_public_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"ok: {SCHEMA_ID}")
        print(f"  provider.base_url = {cfg.provider.base_url}")
        print(f"  provider.model    = {cfg.provider.model}")
        print(f"  provider.api_key_env = {cfg.provider.api_key_env!r}")
        print(f"  parcae_bin_dir    = {cfg.parcae_bin_dir}")
        print(f"  data_dir          = {cfg.data_dir}")
        print(f"  workspace         = {cfg.workspace}")
        print(f"  allow_cuda        = {cfg.allow_cuda}")
        print(
            "  budgets           = "
            f"steps={cfg.budgets.max_steps} "
            f"tools={cfg.budgets.max_tool_calls} "
            f"wall={cfg.budgets.max_wall_seconds}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
