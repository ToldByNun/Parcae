# Architecture

Guides for how the C++ toolkit is structured and how CUDA attaches.

| Doc | Contents |
|-----|----------|
| [cpu-reference.md](cpu-reference.md) | CPU module map, data flow, hot-path rules |
| [cuda-handoff.md](cuda-handoff.md) | CPU exit criteria, twin list, parity hash procedure |
| [cuda-abi.md](cuda-abi.md) | Device buffer shapes, SoA, interrupt encoding |
| [cuda-roadmap.md](cuda-roadmap.md) | **Frozen** CUDA commit roadmap (VS layout) |

**CUDA implementation directory:** [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (Visual Studio).

Specs remain authoritative: [`docs/spec/`](../spec/README.md).
