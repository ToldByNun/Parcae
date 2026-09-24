# Release artifacts

This directory holds **CI-produced** Parcae release files plus tracked notes.
Binaries and per-version staging folders are gitignored.

| Tracked | Role |
|---------|------|
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | Canonical SemVer release notes |
| [`README.md`](README.md) | This file |
| [`.gitignore`](.gitignore) | Local ignore hints |

When a version tag matching `v[0-9]+.*` is pushed (or Release workflow is run
via `workflow_dispatch`), [`.github/workflows/release.yml`](../.github/workflows/release.yml)
assembles:

```text
Parcae-vX.Y.Z-windows-x64-cpu.exe
Parcae-vX.Y.Z-windows-x64-cuda.exe
Parcae-vX.Y.Z-windows-x64.exe

Parcae-vX.Y.Z-linux-x86_64-cpu.tar.gz
Parcae-vX.Y.Z-linux-x86_64-cuda.tar.gz
Parcae-vX.Y.Z-linux-x86_64.tar.gz

Parcae-vX.Y.Z-linux-x86_64-cpu.AppImage
Parcae-vX.Y.Z-linux-x86_64-cuda.AppImage
Parcae-vX.Y.Z-linux-x86_64.AppImage

parcae_X.Y.Z_amd64_cpu.deb
parcae_X.Y.Z_amd64_cuda.deb
parcae_X.Y.Z_amd64.deb

parcae-X.Y.Z-1.x86_64.cpu.rpm
parcae-X.Y.Z-1.x86_64.cuda.rpm
parcae-X.Y.Z-1.x86_64.rpm

parcae-X.Y.Z-source.tar.gz
SHA256SUMS
```

Current cut uses **`X.Y.Z = 0.8.0`** (git tag `v0.8.0-dsl-console`). CUDA-flavored
assets need self-hosted runners — see
[`docs/architecture/release.md`](../docs/architecture/release.md).

Download published files from the GitHub **Releases** page for the tag.
