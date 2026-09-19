# CUDA sources (Visual Studio + optional CMake)

Canonical home for Parcae CUDA twins.

| Build | How |
|-------|-----|
| **Visual Studio** | Open [`Parcae/Parcae.slnx`](../../Parcae.slnx), build **x64**. Requires NVIDIA CUDA Toolkit VS integration (`BuildCustomizations\CUDA <ver>.props`). Default toolkit prop version: **13.3** — override MSBuild property `ParcaeCudaToolkitVersion` if needed. |
| **CMake** | `cmake -S . -B build -DPARCAE_BUILD_CUDA=ON` then build target `parcae_cuda` / alias `parcae::cuda`. Default OFF so CPU CI stays Toolkit-free. |

Step-by-step (VS + CMake + Catch2 tags / skips): [`docs/architecture/cuda-build.md`](../../../docs/architecture/cuda-build.md).

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
| `candidate_batch_buffers.hpp` | `CandidateBatchBuffers` — host SoA ABI v0 (`kMaxC`/`kMaxT`, shared or per-candidate tokens) |
| `backend.hpp` | `CudaBackend` — full catalog `apply_into` / `apply_and_capture` |
| `cuda_score.hpp` | `CudaScore` — `ScoreRegistry` twin (string-id score dispatch) |
| `identity_copy.hpp` / `identity_copy.cu` | `IdentityCopy` — device `uint8_t` identity / smoke kernel |
| `atbash_kernel.hpp` / `atbash_kernel.cu` | `AtbashKernel` — `out[i] = 28 - in[i]` twin |
| `atbash_batch_kernel.hpp` / `.cu` | `AtbashBatchKernel` — shared tokens → atbash lanes (SoA) |
| `atbash_caesar_batch_kernel.hpp` / `.cu` | `AtbashCaesarBatchKernel` — Koan-1 atbash∘caesar 29 shifts |
| `caesar_kernel.hpp` / `caesar_kernel.cu` | `CaesarKernel` — add/sub `shift` mod 29 twin |
| `caesar_batch_kernel.hpp` / `.cu` | `CaesarBatchKernel` — shared tokens + 29 shift lanes (SoA ABI v0) |
| `z29_device.hpp` | `Z29Device` — host/device \\(\\mathbb{Z}_{29}\\) add/mul/inv |
| `affine_kernel.hpp` / `affine_kernel.cu` | `AffineKernel` — `a·x+b` / `inv(a)·(x-b)` twin |
| `affine_batch_kernel.hpp` / `.cu` | `AffineBatchKernel` — shared tokens + 812 `(a,b)` lanes |
| `vigenere_key_kernel.hpp` / `.cu` | `VigenereKeyKernel` — key ring + interrupt skips twin |
| `vigenere_batch_kernel.hpp` / `.cu` | `VigenereBatchKernel` — explicit key-list SoA batch |
| `beaufort_key_kernel.hpp` / `.cu` | `BeaufortKeyKernel` — `key-in` involution + skips twin |
| `totient_prime_stream_kernel.hpp` / `.cu` | `TotientPrimeStreamKernel` — host shifts + skips twin |
| `compose_driver.hpp` / `compose_driver.cu` | `ComposeDriver` — host-orchestrated stages + ping-pong |
| `exact_match_score.hpp` / `.cu` | `ExactMatchScore` — integer mismatch reduce → `0.0`/`1.0` |
| `hamming_agreement_score.hpp` / `.cu` | `HammingAgreementScore` — integer matches → one FP divide |
| `ic_mod29_score.hpp` / `.cu` | `IcMod29Score` — hist 29 bins → fixed-order IC finalize |
| `self_repeat_rate_score.hpp` / `.cu` | `SelfRepeatRateScore` — adjacent-equal edges → one FP divide |
| `chi2_english_gp_score.hpp` / `.cu` | `Chi2EnglishGpScore` — hist + fixed-order χ² finalize |
| `CMakeLists.txt` | Defines `parcae_cuda` when parent enables CUDA |

CPU reference headers remain under repo-root `include/parcae/`.

ABI: [`docs/architecture/cuda-abi.md`](../../../docs/architecture/cuda-abi.md)  
Twins list: [`docs/architecture/cuda-handoff.md`](../../../docs/architecture/cuda-handoff.md)  
Roadmap: [`docs/architecture/cuda-roadmap.md`](../../../docs/architecture/cuda-roadmap.md)
