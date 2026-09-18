# CUDA sources (Visual Studio + optional CMake)

Canonical home for Parcae CUDA twins.

| Build | How |
|-------|-----|
| **Visual Studio** | Open [`Parcae/Parcae.slnx`](../../Parcae.slnx), build **x64**. Requires NVIDIA CUDA Toolkit VS integration (`BuildCustomizations\CUDA <ver>.props`). Default toolkit prop version: **13.3** — override MSBuild property `ParcaeCudaToolkitVersion` if needed. |
| **CMake** | `cmake -S . -B build -DPARCAE_BUILD_CUDA=ON` then build target `parcae_cuda` / alias `parcae::cuda`. Default OFF so CPU CI stays Toolkit-free. |

Files:

| File | Role |
|------|------|
| `parcae_cuda.hpp` | `ParcaeCuda::available` |
| `parcae_cuda_stub.cu` | Minimal TU so the library/project links |
| `cuda_error.hpp` | `CudaError::to_status` — map `cudaError_t` → `Status` |
| `device_buffer.hpp` | RAII `DeviceBuffer<T>` — alloc / H2D / D2H / free |
| `params.hpp` | POD classes: Caesar/Affine/Key/Totient/ComposeStage + CudaDir/CudaFamilyId |
| `params_json.hpp` | `CudaParamsJson` host JSON ↔ POD converters |
| `interrupt_device_view.hpp` | `InterruptDeviceView` — bitmask (`T≤4096`) or sorted `uint32_t` skips |
| `backend.hpp` | `CudaBackend` — `apply_into` / `apply` twin entry (dispatch; kernels later) |
| `identity_copy.hpp` / `identity_copy.cu` | Smoke `IdentityCopy` — device `uint8_t` identity kernel |
| `CMakeLists.txt` | Defines `parcae_cuda` when parent enables CUDA |

CPU reference headers remain under repo-root `include/parcae/`.

ABI: [`docs/architecture/cuda-abi.md`](../../../docs/architecture/cuda-abi.md)  
Twins list: [`docs/architecture/cuda-handoff.md`](../../../docs/architecture/cuda-handoff.md)  
Roadmap: [`docs/architecture/cuda-roadmap.md`](../../../docs/architecture/cuda-roadmap.md)
