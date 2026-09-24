# Spec: DSL AST JSON Frontend Protocol

**Status:** Normative  
**Schema id:** `parcae.dsl_ast_json.v0`  
**`dsl_ast_json_version`:** `1.1.0`  
**Related:** [dsl.md](dsl.md), [theory-artifact.md](theory-artifact.md)

This document freezes the **one-shot CPython → C++ handoff**: a theory `.py`
file is parsed with CPython `ast.parse`, then serialized as JSON. The C++
compiler (`DslAstJsonIngest`) reads only this JSON. Verification, optimization,
fusion, and code emission **MUST NOT** re-enter CPython.

Language rules remain in [dsl.md](dsl.md). This file is the **wire format** only.

---

## Roles

| Component | Duty |
|-----------|------|
| Frontend (`ast_dump`) | Read UTF-8 `.py`, `ast.parse`, emit one JSON document on stdout (or a temp path agreed by CLI) |
| `parcae-compile` | Spawn frontend once; feed JSON to C++ ingest |
| `DslAstJsonIngest` | Enforce limits + schema; build internal `DslAst` |
| `DslSemanticGate` | Apply DSL whitelist (imports, forbidden node kinds, etc.) |

Syntax errors from CPython **MUST** become a structured frontend failure (see
§ Failure envelope), not a C++ crash.

---

## Versioning

| Field | Meaning |
|-------|---------|
| `schema` | Always `parcae.dsl_ast_json.v0` for this document’s shape |
| `dsl_ast_json_version` | SemVer of this protocol (`1.1.0` here) |

Breaking changes to required keys or node shapes **MUST** bump **MAJOR** of
`dsl_ast_json_version` (and typically the `schema` id suffix). Ingestors **MUST**
reject unknown major protocols.

This version is independent of `dsl_spec_version` in [dsl.md](dsl.md) (language
semantics) but both are recorded for diagnostics.

---

## Success document shape

Top-level object:

```json
{
  "schema": "parcae.dsl_ast_json.v0",
  "dsl_ast_json_version": "1.1.0",
  "source_path": "theories/examples/new_math_example.py",
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "python_version": "3.12.0",
  "module": { "...": "Module node — see Node model" },
  "directives": []
}
```

### Required top-level fields

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.dsl_ast_json.v0` |
| `dsl_ast_json_version` | SemVer string |
| `source_path` | Path string as passed to the frontend (may be relative) |
| `source_sha256` | Lowercase hex SHA-256 of the exact UTF-8 source bytes parsed |
| `module` | JSON object for the `Module` AST node |

### Optional top-level fields

| Field | Rule |
|-------|------|
| `python_version` | `sys.version_info` formatted string; SHOULD be present |
| `ok` | `true` on success documents when present |
| `directives` | Array of `#ignore DSL_FLAG:…` entries (see § Directives); **MUST** be present on documents produced by toolchain ≥ `1.1.0` (may be empty). Older `1.0.x` documents omit it. |

Extra unknown top-level keys: conforming ingest **MUST** reject in **strict**
mode (default for `parcae-compile`).

---

## Directives (`#ignore DSL_FLAG`)

CPython `ast` drops `#` comments. The frontend **MUST** collect directives with
the `tokenize` module alongside `ast.parse` and emit them as a top-level
`directives` array (additive in `dsl_ast_json_version` **1.1.0**).

```json
"directives": [
  {
    "lineno": 42,
    "flag": "divergent_branch",
    "raw": "#ignore DSL_FLAG:divergent_branch"
  }
]
```

### Grammar (strict)

```text
COMMENT := '#' [ \t]* 'ignore' [ \t]+ 'DSL_FLAG:' FLAG_NAME [ \t]*
FLAG_NAME := [a-z][a-z0-9_]*
```

Non-matching comments **MUST** be ignored (not emitted). Each emitted object
**MUST** include:

| Field | Rule |
|-------|------|
| `lineno` | 1-based source line of the comment token |
| `flag` | `FLAG_NAME` |
| `raw` | Exact comment token string |

### Recognized flags (v0)

| Flag | Suppresses (when honored — see DirectiveTable / compile) |
|------|----------------------------------------------------------|
| `divergent_branch` | **E033** on the bound HotLoop `if` / `IfExp` |
| `hotloop_restriction` | **E034** for `for`/`while` in HotLoop |
| `host_loop_bound` | **E035** when author asserts a finite OuterControl `while` |

Binding of flags to AST statements and `--allow-dsl-ignores` / **W010** is
specified in [dsl.md](dsl.md) § Execution scopes and implemented by
`DslDirectiveTable` (follow-on to ingest of this array).

---

## Failure envelope

When `ast.parse` fails or the frontend refuses the file, stdout (or the error
channel agreed by CLI) **MUST** be a JSON object:

```json
{
  "schema": "parcae.dsl_ast_json.v0",
  "ok": false,
  "error": {
    "kind": "syntax_error",
    "message": "invalid syntax",
    "lineno": 12,
    "col_offset": 4,
    "end_lineno": 12,
    "end_col_offset": 9
  }
}
```

| Field | Rule |
|-------|------|
| `ok` | MUST be `false` |
| `error.kind` | `syntax_error` \| `io_error` \| `limit_exceeded` \| `encode_error` \| `internal_error` |
| Location fields | Present when known; otherwise null / omitted |

`parcae-compile` **MUST** map this to a user-facing diagnostic and exit non-zero.
It **MUST NOT** treat a failure envelope as a `Module` tree.

Success documents **MUST NOT** set `ok: false`. Success documents MAY omit `ok`
or set `ok: true`.

---

## Ingest limits (normative ceilings)

C++ ingest **MUST** enforce at least these ceilings (values are v0 defaults;
implementations MAY use stricter limits, **MUST NOT** use looser ones without a
spec bump):

| Limit | Default ceiling | On exceed |
|-------|-----------------|-----------|
| Source file size | 1_048_576 bytes (1 MiB) | `limit_exceeded` / `E1xx` |
| JSON document size | 8_388_608 bytes (8 MiB) | same |
| AST node count | 50_000 | same |
| Max tree depth | 64 | same |
| Max string length (any string field) | 16_384 UTF-8 bytes | same |
| Max list length (e.g. `body`, `args`) | 4_096 | same |

Limits exist to harden community submissions (deep nesting, huge literals).
Exceeding a limit is a **diagnostic failure**, never undefined behavior or an
abort without message.

Frontend **SHOULD** pre-check source size before `ast.parse` and emit
`limit_exceeded` itself when possible.

---

## Node model

Every AST node is a JSON object with:

| Field | Rule |
|-------|------|
| `kind` | String discriminant (CPython type name, e.g. `Module`, `ClassDef`) |
| `lineno` | 1-based int; required on nodes that CPython locates |
| `col_offset` | 0-based int |
| `end_lineno` | int or null |
| `end_col_offset` | int or null |

Children are nested objects or arrays of objects. `null` means absent optional
child (same as CPython `None`).

### Location

Ingest **MUST** preserve locations for diagnostics. If a node lacks location in
CPython, fields MAY be null; semantic gate errors then attach to the nearest
ancestor with a location.

---

## Allowed `kind` values (whitelist for DSL sources)

The frontend MAY dump a full module AST. After ingest, `DslSemanticGate`
**MUST** reject any node kind not in the allow-list below (stable `rule_id`,
with lineno). This list is the **DSL-facing** set; it is intentionally smaller
than full Python.

### Structural

`Module`, `ClassDef`, `FunctionDef`, `arguments`, `arg`, `Return`, `Expr`,
`Assign`, `AnnAssign`, `Pass`, `Raise`, `If`, `For`, `While`, `Break`,
`Continue` (scope-conditioned — see [dsl.md](dsl.md) § Execution scopes)

### Imports

`ImportFrom`, `alias` — only; bare `Import` **MUST** be rejected by the gate

### Expressions

`Name`, `Attribute`, `Call`, `Constant`, `BinOp`, `UnaryOp`, `Compare`,
`BoolOp`, `IfExp` (HotLoop → `Z29Expr::Select`; see [dsl.md](dsl.md) § Execution
scopes), `Subscript`, `Tuple`, `List`, `Dict`, `Starred` (only where gate
allows; v0 primitive bodies **MUST NOT** use starred args)

### Operators / ctx (as nested objects or string enums)

Implementations MUST accept either:

- Nested `{ "kind": "Add" }` / `{ "kind": "Load" }`, or
- Compact string `"Add"` / `"Load"` on operator fields

Conforming frontends **SHOULD** use compact strings for ops/ctx to shrink JSON.

Allowed binary ops: `Add`, `Sub`, `Mult`, `Div`, `FloorDiv`, `Mod`, `Pow`,
`LShift`, `RShift`, `BitOr`, `BitXor`, `BitAnd`  
Allowed unary ops: `UAdd`, `USub`, `Not`, `Invert`  
Allowed compare ops: `Eq`, `NotEq`, `Lt`, `LtE`, `Gt`, `GtE`  
Allowed bool ops: `And`, `Or`  
(`MatMult`, `Is`, `IsNot`, `In`, `NotIn` **MUST** be rejected.)  
Allowed bool ops: `And`, `Or`  
Allowed compare ops: `Eq`, `NotEq`, `Lt`, `LtE`, `Gt`, `GtE`, `In`, `NotIn`  
Allowed ctx: `Load`, `Store`

### Decorators

Represented as `decorator_list`: array of expression nodes (`Call`, `Name`,
`Attribute`).

### Explicitly forbidden kinds (non-exhaustive)

Gate **MUST** reject at least: `AsyncFunctionDef`, `Await`, `Yield`,
`YieldFrom`, `Lambda`, `ListComp`, `SetComp`, `DictComp`, `GeneratorExp`,
`With`, `AsyncWith`, `Try`, `ExceptHandler`, `AsyncFor`,
`Global`, `Nonlocal`, `Delete`, `Assert`, `ClassDef` nested inside functions
(v0: classes only at module level), `Import` (non-`ImportFrom`).

Scope-conditioned (not globally forbidden — see [dsl.md](dsl.md) § Execution
scopes): `If`, `For`, `While`, `Break`, `Continue`. OuterControl may use them
under host rules; HotLoop `for` / `while` / `break` / `continue` → **E034**.
HotLoop divergent `if` → **E033** (`DslDivergenceGate`).

---

## Required field shapes (selected kinds)

### `Module`

```json
{
  "kind": "Module",
  "lineno": 1,
  "col_offset": 0,
  "end_lineno": null,
  "end_col_offset": null,
  "body": [ ],
  "type_ignores": []
}
```

### `ImportFrom`

```json
{
  "kind": "ImportFrom",
  "lineno": 3,
  "col_offset": 0,
  "end_lineno": 3,
  "end_col_offset": 40,
  "module": "parcae.dsl.math",
  "names": [
    { "kind": "alias", "name": "z29_add", "asname": null, "lineno": 3, "col_offset": 0, "end_lineno": 3, "end_col_offset": 0 }
  ],
  "level": 0
}
```

`module` MUST be a string; `level` MUST be `0` for DSL sources (relative imports
rejected by gate).

### `ClassDef`

```json
{
  "kind": "ClassDef",
  "name": "MyAffine",
  "bases": [],
  "keywords": [],
  "decorator_list": [],
  "body": [],
  "lineno": 10,
  "col_offset": 0,
  "end_lineno": 40,
  "end_col_offset": 0
}
```

### `FunctionDef`

```json
{
  "kind": "FunctionDef",
  "name": "decrypt_step",
  "args": { "kind": "arguments", "...": "..." },
  "body": [],
  "decorator_list": [],
  "returns": null,
  "lineno": 20,
  "col_offset": 4,
  "end_lineno": 22,
  "end_col_offset": 40
}
```

### `Call`

```json
{
  "kind": "Call",
  "func": { "kind": "Name", "id": "Param", "ctx": "Load", "lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 5 },
  "args": [],
  "keywords": [
    {
      "kind": "keyword",
      "arg": "min",
      "value": { "kind": "Constant", "value": 0, "lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 1 }
    }
  ],
  "lineno": 1,
  "col_offset": 0,
  "end_lineno": 1,
  "end_col_offset": 20
}
```

`keyword` nodes: `arg` is string or `null` (for `**kwargs` — gate rejects
`**kwargs` in v0 DSL).

### `Constant`

```json
{
  "kind": "Constant",
  "value": 28,
  "lineno": 1,
  "col_offset": 0,
  "end_lineno": 1,
  "end_col_offset": 2
}
```

JSON mapping for `value`:

| Python | JSON |
|--------|------|
| `int` | number (integral) |
| `float` | number — gate rejects floats in primitive Z29 bodies |
| `bool` | boolean |
| `str` | string |
| `None` | `null` |
| `bytes` | **MUST** be rejected by frontend or gate in v0 |

### `BinOp`

```json
{
  "kind": "BinOp",
  "left": { "kind": "Name", "id": "c2", "ctx": "Load", "lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 2 },
  "op": "Mult",
  "right": { "kind": "Name", "id": "i", "ctx": "Load", "lineno": 1, "col_offset": 0, "end_lineno": 1, "end_col_offset": 1 },
  "lineno": 1,
  "col_offset": 0,
  "end_lineno": 1,
  "end_col_offset": 8
}
```

### `Name`

```json
{
  "kind": "Name",
  "id": "c0",
  "ctx": "Load",
  "lineno": 1,
  "col_offset": 0,
  "end_lineno": 1,
  "end_col_offset": 2
}
```

### `Attribute` / `Subscript`

Used for `Param[int]`, `self.caesar_shift`, etc. Gate applies DSL rules from
[dsl.md](dsl.md); the JSON **MUST** still carry full location + children.

---

## Frontend algorithm (normative outline)

1. Read file as UTF-8 (**MUST** fail `encode_error` on invalid UTF-8).
2. Enforce source size limit.
3. `tree = ast.parse(source, filename=source_path, type_comments=False)`.
4. Walk tree; serialize only fields listed in this spec (do not emit
   `type_comment` unless a future minor version adds them).
5. Compute `source_sha256` over the exact bytes read.
6. Print a single JSON document (UTF-8), no trailing non-JSON garbage.

Frontends **MUST NOT** execute the module (`exec` / import as plugin).

---

## C++ ingest obligations

`DslAstJsonIngest` **MUST**:

1. Validate UTF-8 JSON; reject trailing garbage.
2. Enforce § Ingest limits.
3. Require `schema` + `dsl_ast_json_version` major compatibility.
4. Build an internal tree without trusting host pointers from Python.
5. On any failure, return `Status` / `DslDiag` — **MUST NOT** throw across the
   CLI boundary uncaught, **MUST NOT** invoke UB on malformed input.

`DslSemanticGate` runs after successful ingest and applies [dsl.md](dsl.md)
rules (imports, subset, tiers, etc.).

---

## Non-goals

- Shipping full CPython `ast` dumps with every possible version-specific field
- Using this JSON as a long-term artifact format for theories (that is
  [theory-artifact.md](theory-artifact.md))
- Replacing IR verification with AST shape checks alone
