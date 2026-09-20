"""Tests for parcae.dsl.ast_dump → parcae.dsl_ast_json.v0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from parcae.dsl.ast_dump import (
    DSL_AST_JSON_VERSION,
    MAX_SOURCE_BYTES,
    SCHEMA_ID,
    document_is_success,
    dump_path,
    dump_source_bytes,
    main,
)


@pytest.mark.ci
def test_dump_simple_module_success(tmp_path: Path) -> None:
    src = tmp_path / "t.py"
    raw = b"from parcae.dsl.math import z29_add\n\nx = z29_add(1, 2)\n"
    src.write_bytes(raw)

    doc = dump_path(src)
    assert document_is_success(doc)
    assert doc["schema"] == SCHEMA_ID
    assert doc["dsl_ast_json_version"] == DSL_AST_JSON_VERSION
    assert doc["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert doc["module"]["kind"] == "Module"
    assert doc["ok"] is True

    # Compact op/ctx strings somewhere in the tree.
    blob = json.dumps(doc)
    assert '"Load"' in blob or "Load" in blob


@pytest.mark.ci
def test_dump_binop_uses_compact_op(tmp_path: Path) -> None:
    src = tmp_path / "poly.py"
    src.write_text(
        "def poly(i, c2):\n    return c2 * i\n",
        encoding="utf-8",
    )
    doc = dump_path(src)
    assert document_is_success(doc)

    def find_binop(node: object) -> dict | None:
        if isinstance(node, dict):
            if node.get("kind") == "BinOp":
                return node
            for v in node.values():
                found = find_binop(v)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = find_binop(item)
                if found is not None:
                    return found
        return None

    binop = find_binop(doc["module"])
    assert binop is not None
    assert binop["op"] == "Mult"


@pytest.mark.ci
def test_syntax_error_envelope(tmp_path: Path) -> None:
    src = tmp_path / "bad.py"
    src.write_text("def broken(\n", encoding="utf-8")
    doc = dump_path(src)
    assert not document_is_success(doc)
    assert doc["ok"] is False
    assert doc["error"]["kind"] == "syntax_error"
    assert "lineno" in doc["error"]


@pytest.mark.ci
def test_source_size_limit() -> None:
    raw = b"x = 1\n" + (b"#" * (MAX_SOURCE_BYTES + 10))
    doc = dump_source_bytes(raw, "huge.py")
    assert doc["ok"] is False
    assert doc["error"]["kind"] == "limit_exceeded"


@pytest.mark.ci
def test_encode_error_invalid_utf8(tmp_path: Path) -> None:
    src = tmp_path / "badenc.py"
    src.write_bytes(b"x = 1\n\xff\xfe\n")
    doc = dump_path(src)
    assert doc["ok"] is False
    assert doc["error"]["kind"] == "encode_error"


@pytest.mark.ci
def test_bytes_constant_rejected(tmp_path: Path) -> None:
    src = tmp_path / "byt.py"
    src.write_text("x = b'abc'\n", encoding="utf-8")
    doc = dump_path(src)
    assert doc["ok"] is False
    assert doc["error"]["kind"] == "internal_error"
    assert "bytes" in doc["error"]["message"].lower()


@pytest.mark.ci
def test_main_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "ok.py"
    good.write_text("a = 1\n", encoding="utf-8")
    assert main([str(good)]) == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["schema"] == SCHEMA_ID
    assert document_is_success(parsed)

    bad = tmp_path / "bad.py"
    bad.write_text("def (\n", encoding="utf-8")
    assert main([str(bad)]) == 1
    fail = json.loads(capsys.readouterr().out)
    assert fail["ok"] is False


@pytest.mark.ci
def test_decorator_and_param_shape(tmp_path: Path) -> None:
    src = tmp_path / "th.py"
    src.write_text(
        "\n".join(
            [
                "from parcae.dsl.theory import Theory, Param",
                "",
                "@Theory(name='t', family='elementwise', tier='A')",
                "class T:",
                "    a: Param[int] = Param(min=1, max=28)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    doc = dump_path(src)
    assert document_is_success(doc)
    classes = [
        n for n in doc["module"]["body"] if isinstance(n, dict) and n.get("kind") == "ClassDef"
    ]
    assert len(classes) == 1
    assert classes[0]["name"] == "T"
    assert classes[0]["decorator_list"]
    assert classes[0]["decorator_list"][0]["kind"] == "Call"
