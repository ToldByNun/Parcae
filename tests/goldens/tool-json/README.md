# Tool JSON envelope goldens

Committed stdout from agent-facing CLIs with `--json` (`parcae.tool_response.v0`).

Regenerate (from repo root, Debug CUDA build tree):

```powershell
$tools = ".\build-cuda\tools\Debug"
$out = "tests\goldens\tool-json"
# keep inputs/mini-runes.txt (UTF-8 bytes, no BOM)
& "$tools\parcae-tokenize.exe" --data-dir data --json "$out\inputs\mini-runes.txt"
# … same invocations as tests/tool_json_golden_test.cpp
```

Compare by parsed JSON equality (not raw string), so key order and
trailing newlines may differ as long as the object matches.
