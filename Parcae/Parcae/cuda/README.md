# CUDA sources (Visual Studio)

This folder is the **canonical home** for Parcae CUDA twins.

- Visual Studio project: [`../Parcae.vcxproj`](../Parcae.vcxproj)
- Solution: [`../../Parcae.slnx`](../../Parcae.slnx)
- ABI freeze: [`docs/architecture/cuda-abi.md`](../../../docs/architecture/cuda-abi.md)
- Twin list / parity procedure: [`docs/architecture/cuda-handoff.md`](../../../docs/architecture/cuda-handoff.md)
- CUDA roadmap: [`docs/architecture/cuda-roadmap.md`](../../../docs/architecture/cuda-roadmap.md)

CPU reference headers remain under repo-root `include/parcae/`.

`.cu` kernels and device helpers are added in later commits (scaffold →
transform twins → scores → batch).
