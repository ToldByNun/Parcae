# Canonical CUDA twin sources for Parcae (Visual Studio + optional CMake).
#
# VS: parent Parcae.vcxproj compiles *.cu here when the CUDA Toolkit build
# customizations are installed.
# CMake: root option PARCAE_BUILD_CUDA=ON builds target parcae::cuda from these files.

set(PARCAE_CUDA_SOURCES
  parcae_cuda_stub.cu
  identity_copy.cu
  atbash_kernel.cu
  atbash_batch_kernel.cu
  atbash_caesar_batch_kernel.cu
  caesar_kernel.cu
  caesar_batch_kernel.cu
  dsl_smoke_caesar_kernel.cu
  affine_kernel.cu
  affine_batch_kernel.cu
  vigenere_key_kernel.cu
  vigenere_batch_kernel.cu
  beaufort_key_kernel.cu
  totient_prime_stream_kernel.cu
  compose_driver.cu
  exact_match_score.cu
  hamming_agreement_score.cu
  ic_mod29_score.cu
  self_repeat_rate_score.cu
  chi2_english_gp_score.cu
  chi2_batch_score.cu
  caesar_chi2_batch.cu
  family_chi2_batch.cu
  deep_score_batch.cu
  z29_matrix_device_ops.cu
)

add_library(parcae_cuda STATIC ${PARCAE_CUDA_SOURCES})
add_library(parcae::cuda ALIAS parcae_cuda)

find_package(CUDAToolkit REQUIRED)

target_include_directories(parcae_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(parcae_cuda
  PUBLIC
    parcae::core
    CUDA::cudart
)

target_compile_features(parcae_cuda PUBLIC cxx_std_20 cuda_std_20)
target_compile_definitions(parcae_cuda PUBLIC PARCAE_HAS_CUDA=1)

# No separable compilation: a static .cu lib with -rdc leaves
# __cudaRegisterLinkedBinary unresolved when linked into a host-only exe
# (parcae_tests). Re-enable with CUDA_RESOLVE_DEVICE_SYMBOLS when kernels need RDC.
set_target_properties(parcae_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Shared
)

# Keep host-side warnings light under nvcc (avoids cl D9025 /W4 vs /W1).
if(MSVC)
  target_compile_options(parcae_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3,/permissive->
  )
else()
  target_compile_options(parcae_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
  )
endif()

message(STATUS "  parcae_cuda: ON (sources in Parcae/Parcae/cuda)")
