# Release packaging

Scripts and Inno/Linux helpers that produce the GitHub Release artifact matrix.
Triggered only by version tags (`v[0-9]+.*`) or manual `workflow_dispatch` —
see [`.github/workflows/release.yml`](../.github/workflows/release.yml).

## Layout

| Path | Role |
|------|------|
| `version.sh` | SemVer from `CMakeLists.txt` `project(Parcae VERSION …)` |
| `stage_payload.sh` | Stage `bin/`, `data/`, `src/`, `python/`, `agents/` |
| `build_source.sh` | `parcae-X.Y.Z-source.tar.gz` via `git archive` |
| `sha256sums.sh` | Write `SHA256SUMS` for a `release/` directory |
| `windows/` | Inno Setup installer + winget bootstrap |
| `linux/` | tar.gz, AppImage, deb, rpm builders |

## Flavors

| Flavor | Binaries |
|--------|----------|
| `cpu` | `PARCAE_BUILD_CUDA=OFF` |
| `cuda` | `PARCAE_BUILD_CUDA=ON` |
| `full` | `bin/` (cpu) + `bin-cuda/` (cuda); unflavored filenames |

## Local smoke (CPU stage only)

```bash
cmake -S . -B build -DPARCAE_BUILD_TOOLS=ON -DPARCAE_BUILD_TESTS=OFF -DPARCAE_BUILD_CUDA=OFF
cmake --build build --config Release
packaging/stage_payload.sh --out /tmp/parcae-stage --flavor cpu --bin-cpu build/tools/Release   # Windows
# or: --bin-cpu build/tools   # Linux/Ninja
packaging/build_source.sh --out /tmp/parcae-release
```

Windows installer additionally needs [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`ISCC.exe`).

Operator guide: [`docs/architecture/release.md`](../docs/architecture/release.md).
