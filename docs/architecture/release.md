# Cutting a Parcae release

**Status:** Packaging + tag-only GitHub Actions workflow  
**Toolkit version source:** [`CMakeLists.txt`](../../CMakeLists.txt) `project(Parcae VERSION …)`  
**Workflow:** [`.github/workflows/release.yml`](../../.github/workflows/release.yml)  
**Scripts:** [`packaging/`](../../packaging/)

Everyday CI ([`ci.yml`](../../.github/workflows/ci.yml)) does **not** publish
installers. Releases run only when you cut a real version tag.

## Artifact names

Filenames use SemVer `X.Y.Z` from CMake (e.g. `0.9.0`). The git tag may carry a
milestone suffix (`v0.9.0-bench`); that suffix is **not** in artifact
names. Human-readable notes:
[`release/0.9.0/RELEASE_NOTES.md`](../../release/0.9.0/RELEASE_NOTES.md).

| Kind | Pattern |
|------|---------|
| Windows installer | `Parcae-vX.Y.Z-windows-x64{,-cpu,-cuda}.exe` |
| Linux tarball | `Parcae-vX.Y.Z-linux-x86_64{,-cpu,-cuda}.tar.gz` |
| AppImage | `Parcae-vX.Y.Z-linux-x86_64{,-cpu,-cuda}.AppImage` |
| Debian | `parcae_X.Y.Z_amd64{_cpu,_cuda,}.deb` |
| RPM | `parcae-X.Y.Z-1.x86_64{.cpu,.cuda,}.rpm` |
| Source | `parcae-X.Y.Z-source.tar.gz` |
| Checksums | `SHA256SUMS` |

Flavors:

- **cpu** — `PARCAE_BUILD_CUDA=OFF` CLIs
- **cuda** — CUDA-linked CLIs (self-hosted GPU runner)
- **unflavored** — preferred “normal” package: cpu+cuda side-by-side when CUDA
  job ran; otherwise cpu-only content under the unflavored name until CUDA
  runners are available

## Windows installer behavior

Inno Setup payload under `%ProgramFiles%\Parcae\`:

- `bin/` (and `bin-cuda/` for full), `data/`, `src/` (full tree for developers),
  `python/`, `agents/`, `LICENSE`, `README.md`
- Start Menu shortcuts + optional desktop icon
- Optional **Add to PATH** (system) for `bin\`
- Optional editable `pip install -e` for `python/` and `agents/`
- Post-install [`packaging/windows/bootstrap.ps1`](../../packaging/windows/bootstrap.ps1)
  checks/installs via **winget**:
  - MSVC C++ Build Tools
  - CMake ≥ 3.25
  - Python ≥ 3.11
  - CUDA Toolkit (cuda / full flavors only; large download, may need reboot)

CUDA Toolkit is **not** bundled inside the `.exe`.

## Linux packages

Prebuilt CLIs + `data/` + sources. `.deb` / `.rpm` declare runtime library
depends and *recommend* cmake / g++ / python3 (and cuda toolkit notes for cuda
flavors). AppImages are portable (no system PATH mutation).

## How to cut a release

1. Bump `project(Parcae VERSION …)` in `CMakeLists.txt` (and matching
   `PARCAE_VERSION_*` fallbacks in `include/parcae/core/version.hpp` if you keep
   them in sync).
2. Land docs/tests for the milestone; keep [`ci.yml`](../../.github/workflows/ci.yml) green.
3. Create an **annotated** tag, e.g. `v0.9.0-bench` (bench / diagnostics exit;
   prior cuts: `v0.8.0-dsl-console`, `v0.7.0-search-engine`).
4. **Push the tag** to GitHub (`git push origin v0.9.0-bench`). The
   Release workflow starts on `v[0-9]+.*` only — not on ordinary commits.
5. Wait for the workflow; confirm assets on the GitHub Release and verify
   `SHA256SUMS`.
6. Optional CUDA matrix: repository variable `PARCAE_HAVE_CUDA_RUNNER=true`
   and self-hosted runners labeled `self-hosted` + `windows|linux` + `cuda`,
   or `workflow_dispatch` with `include_cuda=true`.

Manual re-cut of an existing tag:

```text
Actions → Release → Run workflow → tag = v0.9.0-bench
```

## Local packaging (developers)

See [`packaging/README.md`](../../packaging/README.md). Hosted CI artifacts land
in the workflow’s `release-folder` artifact and on the GitHub Release; the
tracked [`release/`](../../release/) tree keeps per-version `README.md`,
`.gitignore`, and [`RELEASE_NOTES.md`](../../release/0.9.0/RELEASE_NOTES.md).
