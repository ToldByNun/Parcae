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
| `pinned_host_arena.hpp` | RAII `PinnedHostArena` — grow-only `cudaHostAlloc` staging slabs |
| `theory_device_scratch.hpp` | RAII `TheoryDeviceScratch` — persistent buffers; dual param slabs (ping-pong) |
| `cuda_stream_pair.hpp` | RAII `CudaStreamPair` — copy/compute streams + H2D/compute events |
| `params.hpp` | POD classes: Caesar/Affine/Key/Totient/ComposeStage + CudaDir/CudaFamilyId |
| `params_json.hpp` | `CudaParamsJson` host JSON ↔ POD converters |
| `interrupt_device_view.hpp` | `InterruptDeviceView` — bitmask (`T≤4096`) or sorted `uint32_t` skips |
| `interrupt_device_ops.hpp` | `InterruptDeviceOps` — device bitmask / sorted-skip helpers for single-stream kernels |
| `candidate_batch_buffers.hpp` | `CandidateBatchBuffers` — host SoA ABI v0 (`kMaxC`/`kMaxT`, shared or per-candidate tokens) |
| `backend.hpp` | `CudaBackend` — full catalog `apply_into` / `apply_and_capture` |
| `cuda_score.hpp` | `CudaScore` — `ScoreRegistry` twin (string-id score dispatch) |
| `cuda_batch_score.hpp` | `CudaBatchScore` — lane scores via `CudaScore` + host `BatchOrdering` top-k |
| `identity_copy.hpp` / `identity_copy.cu` | `IdentityCopy` — uchar4/`__ldg` identity + `launch_device_async` (host sync wrapper) |
| `atbash_kernel.hpp` / `atbash_kernel.cu` | `AtbashKernel` — `HistFast::dec_atbash` + uchar4/`launch_device_async` |
| `atbash_batch_kernel.hpp` / `.cu` | `AtbashBatchKernel` — shared tokens → atbash lanes (SoA) |
| `atbash_caesar_batch_kernel.hpp` / `.cu` | `AtbashCaesarBatchKernel` — Koan-1 atbash∘caesar 29 shifts |
| `caesar_kernel.hpp` / `caesar_kernel.cu` | `CaesarKernel` — `HistFast` enc/dec + uchar4/`launch_device_async` |
| `caesar_batch_kernel.hpp` / `.cu` | `CaesarBatchKernel` — shared tokens + 29 shift lanes (SoA ABI v0) |
| `z29_device.hpp` | `Z29Device` — host/device \\(\\mathbb{Z}_{29}\\) add/mul/inv |
| `affine_kernel.hpp` / `affine_kernel.cu` | `AffineKernel` — `a·x+b` / `inv(a)·(x-b)` + uchar4/`launch_device_async` |
| `affine_batch_kernel.hpp` / `.cu` | `AffineBatchKernel` — shared tokens + 812 `(a,b)` lanes |
| `vigenere_key_kernel.hpp` / `.cu` | `VigenereKeyKernel` — dense uchar4 / skip path + `launch_device_async` |
| `autokey_ctak_device.hpp` | `AutokeyCtakDevice` — dense CTAK decrypt key/symbol (shared) |
| `ciphertext_autokey_kernel.hpp` / `.cu` | `CiphertextAutokeyKernel` — CTAK materialize; dense decrypt uchar4 + serial encrypt/skips |
| `deep_score_batch.hpp` / `.cu` | `DeepScoreBatch` — fused multi-key / CTAK autokey / n-gram χ² |
| `vigenere_batch_kernel.hpp` / `.cu` | `VigenereBatchKernel` — explicit key-list SoA batch |
| `beaufort_key_kernel.hpp` / `.cu` | `BeaufortKeyKernel` — dense uchar4 / skip path + `launch_device_async` |
| `totient_prime_stream_kernel.hpp` / `.cu` | `TotientPrimeStreamKernel` — host shifts; dense uchar4 / skip + async |
| `compose_driver.hpp` / `compose_driver.cu` | `ComposeDriver` — host-orchestrated stages + ping-pong |
| `exact_match_score.hpp` / `.cu` | `ExactMatchScore` — integer mismatch reduce → `0.0`/`1.0` |
| `hamming_agreement_score.hpp` / `.cu` | `HammingAgreementScore` — integer matches → one FP divide |
| `ic_mod29_score.hpp` / `.cu` | `IcMod29Score` — hist 29 bins → fixed-order IC finalize |
| `self_repeat_rate_score.hpp` / `.cu` | `SelfRepeatRateScore` — adjacent-equal edges → one FP divide |
| `chi2_english_gp_score.hpp` / `.cu` | `Chi2EnglishGpScore` — hist + fixed-order χ² finalize |
| `chi2_batch_score.hpp` / `.cu` | `Chi2BatchScore` — multi-block hist from `out[C·T]` + finalize |
| `hist_fast.hpp` | `HistFast` — Z/29 helpers + warp-private / fat-tile helpers / **thread-local** (research) |
| `hist_fast_parity.hpp` / `.cu` | `HistFastParity` — warp vs local hist golden compare (Catch2) |
| `caesar_chi2_batch.hpp` / `.cu` | `CaesarChi2Batch` — fused Caesar decrypt→χ² (no out materialize) |
| `family_chi2_batch.hpp` / `.cu` | `FamilyChi2Batch` — fused atbash / atbash∘caesar / affine / vigenère χ² |
| `CMakeLists.txt` | Defines `parcae_cuda` when parent enables CUDA |
| [`emitted/`](emitted/) | **Generated** theory-DSL CUDA twins — do not hand-edit (see README there) |

CPU reference headers remain under repo-root `include/parcae/`.

ABI: [`docs/architecture/cuda-abi.md`](../../../docs/architecture/cuda-abi.md)  
Reference: [`docs/architecture/cuda-reference.md`](../../../docs/architecture/cuda-reference.md)  
Twins list: [`docs/architecture/cuda-handoff.md`](../../../docs/architecture/cuda-handoff.md)  
Roadmap: [`docs/architecture/cuda-roadmap.md`](../../../docs/architecture/cuda-roadmap.md)
