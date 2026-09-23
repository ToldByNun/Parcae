# Release artifacts

This directory holds **CI-produced** Parcae release files. Binaries and archives
are gitignored; only this README is tracked.

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

CUDA-flavored assets require self-hosted runners (see
[`docs/architecture/release.md`](../docs/architecture/release.md)). Without them,
cpu + unflavored (cpu content) + source still publish; `CUDA_ARTIFACTS.txt`
explains the gap.

Download published files from the GitHub **Releases** page for the tag (e.g.
`v0.7.0-search-engine`).
