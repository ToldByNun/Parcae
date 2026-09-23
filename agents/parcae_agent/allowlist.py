"""Default agent tool allow-list / deny-list (mirrors AgentPolicy)."""

from __future__ import annotations

ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "tokenize",
        "decode",
        "score",
        "validate",
        "catalog",
        "generate",
        "rank",
        "hypothesis_init",
        "hypothesis_propose",
        "hypothesis_show",
        "hypothesis_list",
        "hypothesis_score",
        "hypothesis_set_status",
        "search_cycle",
    }
)

# CLI basenames the agent MUST NOT invoke (with or without parcae- prefix).
DENIED_BINARIES: frozenset[str] = frozenset(
    {
        "blind-crack",
        "parcae-blind-crack",
        "throughput-tiers",
        "parcae-throughput-tiers",
        "parity",
        "parcae-parity",
        "parity-gen",
        "parcae-parity-gen",
        "search-run",
        "parcae-search-run",
    }
)

# tool name → argv0 stem (no .exe)
TOOL_BINARY: dict[str, str] = {
    "tokenize": "parcae-tokenize",
    "decode": "parcae-decode",
    "score": "parcae-score",
    "validate": "parcae-validate",
    "catalog": "parcae-catalog",
    "generate": "parcae-generate",
    "rank": "parcae-rank",
    "hypothesis_init": "parcae-hypothesis",
    "hypothesis_propose": "parcae-hypothesis",
    "hypothesis_show": "parcae-hypothesis",
    "hypothesis_list": "parcae-hypothesis",
    "hypothesis_score": "parcae-hypothesis",
    "hypothesis_set_status": "parcae-hypothesis",
    "search_cycle": "parcae-search-cycle",
}

HYPOTHESIS_SUBCOMMAND: dict[str, str] = {
    "hypothesis_init": "init",
    "hypothesis_propose": "propose",
    "hypothesis_show": "show",
    "hypothesis_list": "list",
    "hypothesis_score": "score",
    "hypothesis_set_status": "set-status",
}

# Keys the LLM MAY pass per tool. Bridge-owned keys (data_dir, json, …) are
# injected separately and rejected if supplied by the model.
TOOL_ARG_KEYS: dict[str, frozenset[str]] = {
    "tokenize": frozenset({"input", "strict"}),
    "decode": frozenset(
        {
            "input",
            "manifest",
            "transform_json",
            "transform_id",
            "direction",
            "params_json",
            "key_indices",
            "key_latin",
            "skip_indices",
            "shift",
            "backend",
            "rebuild_text",
        }
    ),
    "score": frozenset(
        {
            "input",
            "score_id",
            "params_json",
            "backend",
            "latin",
            "runes",
            "indices",
            "list",
        }
    ),
    "validate": frozenset({"id", "all", "require_locked"}),
    "catalog": frozenset(
        {"all", "transforms", "scores", "generators", "backends"}
    ),
    "generate": frozenset(
        {
            "input",
            "generator_id",
            "direction",
            "params_json",
            "latin",
            "runes",
            "indices",
            "list",
        }
    ),
    "rank": frozenset(
        {
            "candidates",
            "score_id",
            "k",
            "params_json",
            "latin_max",
            "no_latin",
        }
    ),
    "hypothesis_init": frozenset({"id", "title", "method_json", "utc"}),
    "hypothesis_propose": frozenset(
        {
            "id",
            "method_json",
            "method_file",
            "title",
            "rationale",
            "source_json",
            "utc",
        }
    ),
    "hypothesis_show": frozenset({"id"}),
    "hypothesis_list": frozenset(),
    "hypothesis_score": frozenset(
        {"id", "input", "score_id", "latin", "runes", "indices", "utc"}
    ),
    "hypothesis_set_status": frozenset({"id", "status", "utc"}),
    # workspace / data_dir / json / allow_cuda / omit-timing / quiet injected
    # by ToolBridge. status=true → readiness only (--status); else family|job.
    "search_cycle": frozenset(
        {
            "status",
            "job",
            "family",
            "k",
            "seed",
            "score_id",
            "backend",
            "iterations",
            "created_utc",
            "allow_extended_families",
            "allow_theory_uri",
        }
    ),
}

# Boolean CLI flags (presence = true).
BOOL_FLAGS: dict[str, str] = {
    "strict": "--strict",
    "rebuild_text": "--rebuild-text",
    "latin": "--latin",
    "runes": "--runes",
    "indices": "--indices",
    "list": "--list",
    "all": "--all",
    "require_locked": "--require-locked",
    "transforms": "--transforms",
    "scores": "--scores",
    "generators": "--generators",
    "backends": "--backends",
    "no_latin": "--no-latin",
    "allow_extended_families": "--allow-extended-families",
    "allow_theory_uri": "--allow-theory-uri",
}

# String/int option flags (key → --flag).
VALUE_FLAGS: dict[str, str] = {
    "input": "--input",
    "manifest": "--manifest",
    "transform_json": "--transform-json",
    "transform_id": "--transform-id",
    "direction": "--direction",
    "params_json": "--params-json",
    "key_indices": "--key-indices",
    "key_latin": "--key-latin",
    "skip_indices": "--skip-indices",
    "shift": "--shift",
    "backend": "--backend",
    "score_id": "--score-id",
    "id": "--id",
    "generator_id": "--generator-id",
    "candidates": "--candidates",
    "k": "--k",
    "latin_max": "--latin-max",
    "title": "--title",
    "method_json": "--method-json",
    "method_file": "--method-file",
    "rationale": "--rationale",
    "source_json": "--source-json",
    "utc": "--utc",
    "status": "--status",
    "job": "--job",
    "family": "--family",
    "seed": "--seed",
    "iterations": "--iterations",
    "created_utc": "--created-utc",
}

# Keys the model MUST NOT supply (bridge injects them).
BRIDGE_OWNED_KEYS: frozenset[str] = frozenset(
    {
        "data_dir",
        "json",
        "allow_cuda",
        "workspace",
        "argv",
        "command",
        "shell",
        "executable",
        "bin",
        "cwd",
    }
)
