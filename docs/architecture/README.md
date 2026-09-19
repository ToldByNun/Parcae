# Architecture

Guides for how the C++ toolkit is structured and how CUDA attaches.

| Doc | Contents |
|-----|----------|
| [cpu-reference.md](cpu-reference.md) | CPU module map, data flow, hot-path rules |
| [cuda-reference.md](cuda-reference.md) | CUDA twin layout, batch/fused search, exit checklist |
| [cuda-handoff.md](cuda-handoff.md) | CPU exit criteria, twin list, parity hash procedure |
| [cuda-abi.md](cuda-abi.md) | Device buffer shapes, SoA, interrupt encoding |
| [cuda-build.md](cuda-build.md) | Local CUDA Toolkit / CMake notes; CI stays CPU-default |
| [cuda-throughput.md](cuda-throughput.md) | Fused χ² throughput ceilings (`parcae-throughput-tiers`) |
| [cuda-score-reduction.md](cuda-score-reduction.md) | Score FP / histogram reduction associativity for CUDA twins |
| [cuda-roadmap.md](cuda-roadmap.md) | **Frozen** CUDA commit roadmap (VS layout) |

**CUDA implementation directory:** [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (Visual Studio).

Specs remain authoritative: [`docs/spec/`](../spec/README.md).
