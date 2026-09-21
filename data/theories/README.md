# Theory artifacts (`data/theories/`)

Compiled outputs of `parcae-compile` live here. Normative layout and manifest:
[`docs/spec/theory-artifact.md`](../../docs/spec/theory-artifact.md).
Language / versioning: [`docs/spec/dsl.md`](../../docs/spec/dsl.md).

```text
data/theories/
  <name>/
    <version>/           # e.g. 1
      manifest.json      # parcae.theory_artifact.v0
      …
```

URI form: `parcae://theories/<name>@<version>`.

## Git policy

Runtime compile products under this tree are **gitignored** (see repo
`.gitignore`). This README is kept so the directory exists in a fresh clone.

Committed golden/example artifacts (if any) must be allow-listed explicitly in
`.gitignore` — do not rely on silent exceptions.

## Compile

`parcae-compile <theory.py>` runs ast_dump → ingest → gate → IR → verify → emit and
writes `manifest.json` under this tree. `--status` reports `pipeline_ready`.

