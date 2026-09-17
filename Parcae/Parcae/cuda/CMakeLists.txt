# Canonical CUDA twin sources for Parcae (Visual Studio + optional CMake).
#
# VS: parent Parcae.vcxproj compiles *.cu here when the CUDA Toolkit build
# customizations are installed.
# CMake: root option PARCAE_BUILD_CUDA=ON builds target parcae::cuda from these files.

set(PARCAE_CUDA_SOURCES
  parcae_cuda_stub.cu
)

add_library(parcae_cuda STATIC ${PARCAE_CUDA_SOURCES})
add_library(parcae::cuda ALIAS parcae_cuda)

target_include_directories(parcae_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(parcae_cuda PUBLIC parcae::core)

target_compile_features(parcae_cuda PUBLIC cxx_std_20)
target_compile_definitions(parcae_cuda PUBLIC PARCAE_HAS_CUDA=1)

set_target_properties(parcae_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RUNTIME_LIBRARY Static
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
