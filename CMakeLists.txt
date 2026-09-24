cmake_minimum_required(VERSION 3.25)

project(Parcae
  VERSION 0.8.0
  DESCRIPTION "C++20 Liber Primus / Cicada 3301 cryptanalysis toolkit"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PARCAE_BUILD_TESTS "Build Parcae unit tests" ON)
option(PARCAE_BUILD_TOOLS "Build Parcae CLI tools" ON)
option(PARCAE_BUILD_CUDA "Build CUDA twins from Parcae/Parcae/cuda" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/CompilerWarnings.cmake)
include(cmake/Dependencies.cmake)

# Header-only core for now; evolve to STATIC when non-header sources appear.
# Warning flags attach to real STATIC/OBJECT/EXECUTABLE targets via
# parcae_set_project_warnings(<target>).
add_library(parcae_core INTERFACE)
add_library(parcae::core ALIAS parcae_core)

target_include_directories(parcae_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(parcae_core INTERFACE cxx_std_20)

target_compile_definitions(parcae_core INTERFACE
  PARCAE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  PARCAE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  PARCAE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  PARCAE_VERSION_STRING="${PROJECT_VERSION}"
)

target_link_libraries(parcae_core INTERFACE nlohmann_json::nlohmann_json)

set(PARCAE_HAS_CUDA_BUILD FALSE)
if(PARCAE_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      # 75/86/89 for Turing–Ada; 120 for Blackwell (RTX 50-series).
      set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120")
    endif()
    add_subdirectory(Parcae/Parcae/cuda)
    set(PARCAE_HAS_CUDA_BUILD TRUE)
  else()
    message(FATAL_ERROR
      "PARCAE_BUILD_CUDA=ON but no CUDA compiler was found. "
      "Install the NVIDIA CUDA Toolkit or set CMAKE_CUDA_COMPILER.")
  endif()
endif()

if(PARCAE_BUILD_TESTS OR PARCAE_BUILD_TOOLS)
  enable_testing()
endif()

if(PARCAE_BUILD_TOOLS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
  endif()
endif()

if(PARCAE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

message(STATUS "Parcae ${PROJECT_VERSION}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Catch2: ${PARCAE_CATCH2_VERSION}")
message(STATUS "  nlohmann_json: ${PARCAE_NLOHMANN_JSON_VERSION}")
message(STATUS "  DSL headers: include/parcae/dsl (via parcae::core)")
message(STATUS "  PARCAE_BUILD_TESTS=${PARCAE_BUILD_TESTS}")
message(STATUS "  PARCAE_BUILD_TOOLS=${PARCAE_BUILD_TOOLS}")
message(STATUS "  PARCAE_BUILD_CUDA=${PARCAE_BUILD_CUDA} (active=${PARCAE_HAS_CUDA_BUILD})")
