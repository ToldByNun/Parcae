"""OpenAI-compatible tool/function schemas for allow-listed Parcae tools.

These definitions are what `parcae-agent` passes as `tools` to
`POST /v1/chat/completions`. Property names MUST match `TOOL_ARG_KEYS` in
`allowlist.py` (bridge-owned keys like `data_dir` / `workspace` are omitted).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from parcae_agent.allowlist import ALLOWED_TOOLS, TOOL_ARG_KEYS

# Stable order for prompts / diffs (matches docs/spec/agent-tools.md table).
TOOL_SCHEMA_ORDER: tuple[str, ...] = (
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
)

_BOOL = {"type": "boolean"}
_STRING = {"type": "string", "minLength": 1}
_INT = {"type": "integer"}
_BACKEND = {"type": "string", "enum": ["cpu", "cuda"]}
_DIRECTION = {"type": "string", "enum": ["decrypt", "encrypt"]}
# Object or already-serialized JSON text (ToolBridge accepts both).
_JSONISH: dict[str, Any] = {
    "description": "JSON object or a JSON text string",
    "oneOf": [
        {"type": "object"},
        {"type": "string", "minLength": 1},
    ],
}


def _object(
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    if description:
        schema["description"] = description
    return schema


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


_TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "tokenize": _tool(
        "tokenize",
        "Tokenize UTF-8 Liber Primus text into tokens / Index29 consumables. "
        "Pass input as a filesystem path or '-' for stdin.",
        _object(
            {
                "input": {
                    **_STRING,
                    "description": "Path to UTF-8 text, or '-' for stdin",
                },
                "strict": {
                    **_BOOL,
                    "description": "Reject unknown symbols (default true)",
                },
            },
            required=["input"],
        ),
    ),
    "decode": _tool(
        "decode",
        "Apply a catalog transform (or fixture manifest / transform JSON file) "
        "to ciphertext. Prefer transform_id values from catalog; do not invent ids.",
        _object(
            {
                "input": {**_STRING, "description": "Ciphertext path or '-'"},
                "manifest": {
                    **_STRING,
                    "description": "Fixture directory or manifest.json path",
                },
                "transform_json": {
                    **_STRING,
                    "description": "Path to a transform envelope JSON file",
                },
                "transform_id": {
                    **_STRING,
                    "description": "Catalog transform id (e.g. atbash, caesar)",
                },
                "direction": {
                    **_DIRECTION,
                    "description": "encrypt or decrypt (default decrypt)",
                },
                "params_json": _JSONISH,
                "key_indices": {
                    **_STRING,
                    "description": "Comma-separated Index29 key values 0..28",
                },
                "key_latin": {**_STRING, "description": "Optional key as Latin letters"},
                "skip_indices": {
                    **_STRING,
                    "description": "Comma-separated consumable skip indices",
                },
                "shift": {
                    "description": "Caesar shift shortcut",
                    "oneOf": [{"type": "integer"}, {"type": "string", "minLength": 1}],
                },
                "backend": {
                    **_BACKEND,
                    "description": "cpu (default) or cuda (needs allow_cuda in config)",
                },
                "rebuild_text": {
                    **_BOOL,
                    "description": "Also rebuild UTF-8 with separators preserved",
                },
            }
        ),
    ),
    "score": _tool(
        "score",
        "Score an Index29 / latin / rune stream with a catalog score_id. "
        "Use catalog to discover score ids; do not invent them.",
        _object(
            {
                "input": {**_STRING, "description": "Input path or '-'"},
                "score_id": {
                    **_STRING,
                    "description": "Registry score id (e.g. ic_mod29)",
                },
                "params_json": _JSONISH,
                "backend": {**_BACKEND},
                "latin": {**_BOOL, "description": "Treat input as Latin letters"},
                "runes": {**_BOOL, "description": "Tokenize as Liber Primus runes"},
                "indices": {
                    **_BOOL,
                    "description": "Parse input as Index29 values 0..28",
                },
                "list": {**_BOOL, "description": "List known score ids and exit"},
            }
        ),
    ),
    "validate": _tool(
        "validate",
        "Validate a solved fixture (or all fixtures) against locked oracles.",
        _object(
            {
                "id": {
                    **_STRING,
                    "description": "Fixture id or fixture directory path",
                },
                "all": {**_BOOL, "description": "Validate all solved fixtures"},
                "require_locked": {
                    **_BOOL,
                    "description": "Require verification.status=locked",
                },
            }
        ),
    ),
    "catalog": _tool(
        "catalog",
        "List catalog ids: transforms, scores, generators, and/or backends. "
        "Call this before inventing transform_id / score_id / generator_id.",
        _object(
            {
                "all": {**_BOOL, "description": "All sections (default if none set)"},
                "transforms": {**_BOOL},
                "scores": {**_BOOL},
                "generators": {**_BOOL},
                "backends": {**_BOOL},
            }
        ),
    ),
    "generate": _tool(
        "generate",
        "Emit TransformCandidate JSON from a catalog generator (gen_*). "
        "Discover generator_id via catalog; do not invent generators.",
        _object(
            {
                "input": {**_STRING, "description": "Source path or '-'"},
                "generator_id": {
                    **_STRING,
                    "description": "Generator id (e.g. gen_caesar, gen_atbash)",
                },
                "direction": {**_DIRECTION},
                "params_json": _JSONISH,
                "latin": {**_BOOL},
                "runes": {**_BOOL},
                "indices": {**_BOOL},
                "list": {**_BOOL, "description": "List known generator ids"},
            }
        ),
    ),
    "rank": _tool(
        "rank",
        "Score and rank TransformCandidate JSON (from generate) with a catalog "
        "score_id; returns top-k with stable ties.",
        _object(
            {
                "candidates": {
                    **_STRING,
                    "description": "Candidates JSON path or '-'",
                },
                "score_id": {**_STRING, "description": "Catalog score id"},
                "k": {
                    **_INT,
                    "minimum": 1,
                    "description": "Top-k hits to keep",
                },
                "params_json": _JSONISH,
                "latin_max": {
                    **_INT,
                    "minimum": 0,
                    "description": "Latin preview length (0 = full)",
                },
                "no_latin": {**_BOOL, "description": "Omit latin preview"},
            },
            required=["candidates", "score_id", "k"],
        ),
    ),
    "hypothesis_init": _tool(
        "hypothesis_init",
        "Create a draft HypothesisRecord stub in the configured workspace "
        "(workspace id is injected by ToolBridge; do not pass it).",
        _object(
            {
                "id": {
                    **_STRING,
                    "description": "Hypothesis id (lowercase / _ / -)",
                },
                "title": {**_STRING},
                "method_json": _JSONISH,
                "utc": {**_STRING, "description": "RFC3339 timestamp"},
            },
            required=["id"],
        ),
    ),
    "hypothesis_propose": _tool(
        "hypothesis_propose",
        "Write or update a hypothesis method/rationale in the workspace.",
        _object(
            {
                "id": {**_STRING},
                "method_json": _JSONISH,
                "method_file": {
                    **_STRING,
                    "description": "Path to method JSON file",
                },
                "title": {**_STRING},
                "rationale": {**_STRING},
                "source_json": _JSONISH,
                "utc": {**_STRING},
            },
            required=["id"],
        ),
    ),
    "hypothesis_show": _tool(
        "hypothesis_show",
        "Read one HypothesisRecord from the configured workspace.",
        _object({"id": {**_STRING}}, required=["id"]),
    ),
    "hypothesis_list": _tool(
        "hypothesis_list",
        "List hypotheses in the configured workspace.",
        _object({}),
    ),
    "hypothesis_score": _tool(
        "hypothesis_score",
        "Score a stored hypothesis method against an input stream.",
        _object(
            {
                "id": {**_STRING},
                "input": {**_STRING, "description": "Input path or '-'"},
                "score_id": {**_STRING},
                "latin": {**_BOOL},
                "runes": {**_BOOL},
                "indices": {**_BOOL},
                "utc": {**_STRING},
            },
            required=["id", "input"],
        ),
    ),
    "hypothesis_set_status": _tool(
        "hypothesis_set_status",
        "Update hypothesis status (draft / proposed / rejected / promoted / …).",
        _object(
            {
                "id": {**_STRING},
                "status": {
                    **_STRING,
                    "description": "New status string accepted by parcae-hypothesis",
                },
                "utc": {**_STRING},
            },
            required=["id", "status"],
        ),
    ),
}


def assert_schemas_match_allowlist() -> None:
    """Raise AssertionError if schema set drifts from TOOL_ARG_KEYS."""
    if set(TOOL_SCHEMA_ORDER) != ALLOWED_TOOLS:
        missing = ALLOWED_TOOLS - set(TOOL_SCHEMA_ORDER)
        extra = set(TOOL_SCHEMA_ORDER) - ALLOWED_TOOLS
        raise AssertionError(
            f"TOOL_SCHEMA_ORDER drift: missing={sorted(missing)} extra={sorted(extra)}"
        )
    if set(_TOOL_DEFINITIONS) != ALLOWED_TOOLS:
        raise AssertionError("tool schema names must equal ALLOWED_TOOLS")
    for name in TOOL_SCHEMA_ORDER:
        props = set(
            _TOOL_DEFINITIONS[name]["function"]["parameters"].get("properties", {})
        )
        expected = set(TOOL_ARG_KEYS[name])
        if props != expected:
            raise AssertionError(
                f"{name} schema properties {sorted(props)} != "
                f"TOOL_ARG_KEYS {sorted(expected)}"
            )


def tool_schema(name: str) -> dict[str, Any]:
    """Return one OpenAI tool definition (deep-ish copy via dict rebuild)."""
    if name not in _TOOL_DEFINITIONS:
        raise KeyError(f"unknown tool schema: {name}")
    # Shallow copy is enough: callers must not mutate nested dicts in place.
    import copy

    return copy.deepcopy(_TOOL_DEFINITIONS[name])


def openai_tools(
    names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """OpenAI `tools` array for chat.completions (default: full allow-list)."""
    if names is None:
        ordered = list(TOOL_SCHEMA_ORDER)
    else:
        ordered = list(names)
        unknown = [n for n in ordered if n not in ALLOWED_TOOLS]
        if unknown:
            raise KeyError(f"tools not on allow-list: {unknown}")
    return [tool_schema(n) for n in ordered]


def tool_names() -> tuple[str, ...]:
    return TOOL_SCHEMA_ORDER


# Fail fast on import if schemas drift from the allow-list.
assert_schemas_match_allowlist()
