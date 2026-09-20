"""`parcae-agent` command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parcae_agent import SCHEMA_ID, __version__
from parcae_agent.cli import cmd_doctor, cmd_providers_test, cmd_run
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
    check.add_argument("path", type=Path, help="Path to agent config file")
    check.add_argument(
        "--json",
        action="store_true",
        help="Print the public (secret-free) config as JSON",
    )

    run = sub.add_parser("run", help="Run the Liber Primus agent loop")
    run.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Agent config (parcae.agent_config.v0)",
    )
    run.add_argument("--prompt", "-p", type=str, default=None, help="User prompt text")
    run.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read user prompt from a UTF-8 file",
    )
    run.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable run summary on stdout",
    )
    run.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log tool steps to stderr",
    )
    run.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default 0)",
    )

    providers = sub.add_parser("providers", help="Provider helpers")
    providers_sub = providers.add_subparsers(dest="providers_command", required=True)
    providers_test = providers_sub.add_parser(
        "test",
        help="Send a tiny chat completion to verify the provider",
    )
    providers_test.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Agent config (parcae.agent_config.v0)",
    )
    providers_test.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result",
    )

    doctor = sub.add_parser(
        "doctor",
        help="Check config, data_dir, binaries, and API key env",
    )
    doctor.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Agent config (parcae.agent_config.v0)",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Print JSON checklist",
    )

    args = parser.parse_args(argv)

    if args.command == "check-config":
        return _cmd_check_config(args.path, as_json=args.json)
    if args.command == "run":
        return cmd_run(
            args.config,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            as_json=args.json,
            verbose=args.verbose,
            temperature=args.temperature,
        )
    if args.command == "providers":
        if args.providers_command == "test":
            return cmd_providers_test(args.config, as_json=args.json)
        parser.error(f"unknown providers command: {args.providers_command}")
    if args.command == "doctor":
        return cmd_doctor(args.config, as_json=args.json)

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
