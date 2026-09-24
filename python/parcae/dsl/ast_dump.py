"""CPython AST → parcae.dsl_ast_json.v0 frontend (syntax only; no exec).

Normative: docs/spec/dsl-ast-json.md
Usage: python -m parcae.dsl.ast_dump <theory.py>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_ID = "parcae.dsl_ast_json.v0"
DSL_AST_JSON_VERSION = "1.1.0"

# Normative v0 ceilings (docs/spec/dsl-ast-json.md).
MAX_SOURCE_BYTES = 1_048_576
MAX_JSON_BYTES = 8_388_608
MAX_NODE_COUNT = 50_000
MAX_TREE_DEPTH = 64
MAX_STRING_BYTES = 16_384
MAX_LIST_LENGTH = 4_096

# docs/spec/dsl-ast-json.md § Directives — COMMENT := '#' [ \t]* 'ignore' …
_DIRECTIVE_RE = re.compile(r"^#[ \t]*ignore[ \t]+DSL_FLAG:([a-z][a-z0-9_]*)[ \t]*$")

# Recognized flags (v0); unknown grammar-valid flags are still emitted for tooling.
KNOWN_DSL_FLAGS = frozenset(
    {
        "divergent_branch",
        "hotloop_restriction",
        "host_loop_bound",
    }
)

_COMPACT_NODE_TYPES = (
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.expr_context,
)


class AstDumpError(Exception):
    """Frontend failure with structured envelope fields."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        lineno: int | None = None,
        col_offset: int | None = None,
        end_lineno: int | None = None,
        end_col_offset: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset


@dataclass
class _SerializeState:
    node_count: int = 0
    string_checks: list[str] = field(default_factory=list)


def _python_version_string() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _failure_document(err: AstDumpError) -> dict[str, Any]:
    error: dict[str, Any] = {
        "kind": err.kind,
        "message": err.message,
    }
    if err.lineno is not None:
        error["lineno"] = err.lineno
    if err.col_offset is not None:
        error["col_offset"] = err.col_offset
    if err.end_lineno is not None:
        error["end_lineno"] = err.end_lineno
    if err.end_col_offset is not None:
        error["end_col_offset"] = err.end_col_offset
    return {
        "schema": SCHEMA_ID,
        "ok": False,
        "error": error,
    }


def _check_string(value: str, state: _SerializeState) -> str:
    nbytes = len(value.encode("utf-8"))
    if nbytes > MAX_STRING_BYTES:
        raise AstDumpError(
            "limit_exceeded",
            f"string field exceeds {MAX_STRING_BYTES} UTF-8 bytes ({nbytes})",
        )
    return value


def _location_fields(node: ast.AST) -> dict[str, Any]:
    return {
        "lineno": getattr(node, "lineno", None),
        "col_offset": getattr(node, "col_offset", None),
        "end_lineno": getattr(node, "end_lineno", None),
        "end_col_offset": getattr(node, "end_col_offset", None),
    }


def _serialize(node: Any, state: _SerializeState, depth: int) -> Any:
    if depth > MAX_TREE_DEPTH:
        raise AstDumpError(
            "limit_exceeded",
            f"AST depth exceeds {MAX_TREE_DEPTH}",
        )

    if node is None:
        return None

    if isinstance(node, list):
        if len(node) > MAX_LIST_LENGTH:
            raise AstDumpError(
                "limit_exceeded",
                f"list length exceeds {MAX_LIST_LENGTH} ({len(node)})",
            )
        return [_serialize(item, state, depth + 1) for item in node]

    if isinstance(node, _COMPACT_NODE_TYPES):
        return type(node).__name__

    if isinstance(node, ast.AST):
        state.node_count += 1
        if state.node_count > MAX_NODE_COUNT:
            raise AstDumpError(
                "limit_exceeded",
                f"AST node count exceeds {MAX_NODE_COUNT}",
            )

        kind = type(node).__name__
        out: dict[str, Any] = {"kind": kind, **_location_fields(node)}

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bytes):
                raise AstDumpError(
                    "internal_error",
                    "Constant bytes literals are not supported in dsl_ast_json v0",
                    lineno=getattr(node, "lineno", None),
                    col_offset=getattr(node, "col_offset", None),
                    end_lineno=getattr(node, "end_lineno", None),
                    end_col_offset=getattr(node, "end_col_offset", None),
                )
            if isinstance(value, str):
                out["value"] = _check_string(value, state)
            elif isinstance(value, (int, float, bool)) or value is None:
                out["value"] = value
            else:
                # complex, ellipsis, etc. — keep JSON-safe via string fallback reject
                raise AstDumpError(
                    "internal_error",
                    f"Unsupported Constant type {type(value).__name__} in dsl_ast_json v0",
                    lineno=getattr(node, "lineno", None),
                    col_offset=getattr(node, "col_offset", None),
                )
            return out

        for field_name in node._fields:
            if field_name == "type_comment":
                continue
            raw = getattr(node, field_name, None)
            if isinstance(raw, str):
                out[field_name] = _check_string(raw, state)
            else:
                out[field_name] = _serialize(raw, state, depth + 1)
        return out

    if isinstance(node, (int, float, bool)):
        return node

    if isinstance(node, str):
        return _check_string(node, state)

    # ast.arguments and others shouldn't hit bare unknowns often.
    raise AstDumpError(
        "internal_error",
        f"Unsupported AST payload type {type(node).__name__}",
    )


def extract_directives(source: str) -> list[dict[str, Any]]:
    """Collect `#ignore DSL_FLAG:…` comments via tokenize (ast drops comments).

    Non-matching comments are ignored. Grammar is strict (see dsl-ast-json.md).
    """
    out: list[dict[str, Any]] = []
    readline = io.StringIO(source).readline
    try:
        for tok in tokenize.generate_tokens(readline):
            if tok.type != tokenize.COMMENT:
                continue
            match = _DIRECTIVE_RE.match(tok.string)
            if match is None:
                continue
            out.append(
                {
                    "lineno": tok.start[0],
                    "flag": match.group(1),
                    "raw": tok.string,
                }
            )
    except tokenize.TokenError:
        # Incomplete tokenize; ast.parse already validated syntax for success path.
        return out
    return out


def dump_source_bytes(raw: bytes, source_path: str) -> dict[str, Any]:
    """Parse UTF-8 source bytes and return a success or failure document."""
    if len(raw) > MAX_SOURCE_BYTES:
        return _failure_document(
            AstDumpError(
                "limit_exceeded",
                f"source file exceeds {MAX_SOURCE_BYTES} bytes ({len(raw)})",
            )
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _failure_document(
            AstDumpError(
                "encode_error",
                f"source is not valid UTF-8: {exc}",
            )
        )

    digest = hashlib.sha256(raw).hexdigest()

    try:
        tree = ast.parse(text, filename=source_path, type_comments=False)
    except SyntaxError as exc:
        return _failure_document(
            AstDumpError(
                "syntax_error",
                exc.msg or "invalid syntax",
                lineno=exc.lineno,
                col_offset=(exc.offset - 1) if exc.offset is not None else None,
                end_lineno=getattr(exc, "end_lineno", None),
                end_col_offset=(
                    (exc.end_offset - 1)
                    if getattr(exc, "end_offset", None) is not None
                    else None
                ),
            )
        )

    state = _SerializeState()
    try:
        module_json = _serialize(tree, state, depth=0)
    except AstDumpError as err:
        return _failure_document(err)

    if not isinstance(module_json, dict) or module_json.get("kind") != "Module":
        return _failure_document(
            AstDumpError("internal_error", "ast.parse did not yield a Module node")
        )

    # Spec Module example includes type_ignores; ensure key exists.
    module_json.setdefault("type_ignores", [])

    directives = extract_directives(text)

    doc: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "dsl_ast_json_version": DSL_AST_JSON_VERSION,
        "source_path": source_path,
        "source_sha256": digest,
        "python_version": _python_version_string(),
        "module": module_json,
        "directives": directives,
        "ok": True,
    }

    encoded = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_JSON_BYTES:
        return _failure_document(
            AstDumpError(
                "limit_exceeded",
                f"JSON document exceeds {MAX_JSON_BYTES} bytes ({len(encoded)})",
            )
        )
    return doc


def dump_path(path: str | Path) -> dict[str, Any]:
    """Read a file and dump AST JSON (success or failure document)."""
    source_path = str(path)
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return _failure_document(
            AstDumpError("io_error", f"failed to read source: {exc}")
        )
    return dump_source_bytes(raw, source_path)


def document_is_success(doc: dict[str, Any]) -> bool:
    return doc.get("ok", True) is not False and "module" in doc and "error" not in doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m parcae.dsl.ast_dump",
        description="Dump theory .py AST as parcae.dsl_ast_json.v0 (no exec).",
    )
    parser.add_argument(
        "source",
        help="Path to a theory .py file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write JSON to this path instead of stdout",
    )
    args = parser.parse_args(argv)

    doc = dump_path(args.source)
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    text_out = text + "\n"

    if args.output:
        try:
            Path(args.output).write_text(text_out, encoding="utf-8")
        except OSError as exc:
            fail = _failure_document(
                AstDumpError("io_error", f"failed to write output: {exc}")
            )
            sys.stdout.write(json.dumps(fail, ensure_ascii=False, indent=2) + "\n")
            return 1
    else:
        sys.stdout.write(text_out)

    return 0 if document_is_success(doc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
