# CUDA sources (Visual Studio + optional CMake)

Canonical home for Parcae CUDA twins.

| Build | How |
|-------|-----|
| **Visual Studio** | Open [`Parcae/Parcae.slnx`](../../Parcae.slnx), build **x64**. Requires NVIDIA CUDA Toolkit VS integration (`BuildCustomizations\CUDA <ver>.props`). Default toolkit prop version: **13.3** — override MSBuild property `ParcaeCudaToolkitVersion` if needed. |
| **CMake** | `cmake -S . -B build -DPARCAE_BUILD_CUDA=ON` then build target `parcae_cuda` / alias `parcae::cuda`. Default OFF so CPU CI stays Toolkit-free. |

Files:

| File | Role |
|------|------|
| `parcae_cuda.hpp` | Public façade (`parcae::cuda::available`) |
| `parcae_cuda_stub.cu` | Minimal TU so the library/project links |
| `CMakeLists.txt` | Defines `parcae_cuda` when parent enables CUDA |

CPU reference headers remain under repo-root `include/parcae/`.

ABI: [`docs/architecture/cuda-abi.md`](../../../docs/architecture/cuda-abi.md)  
Twins list: [`docs/architecture/cuda-handoff.md`](../../../docs/architecture/cuda-handoff.md)  
Roadmap: [`docs/architecture/cuda-roadmap.md`](../../../docs/architecture/cuda-roadmap.md)
