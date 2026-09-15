cmake_minimum_required(VERSION 3.25)

project(Parcae
  VERSION 0.1.0
  DESCRIPTION "C++20 Liber Primus / Cicada 3301 cryptanalysis toolkit"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PARCAE_BUILD_TESTS "Build Parcae unit tests" ON)
option(PARCAE_BUILD_TOOLS "Build Parcae CLI tools" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/CompilerWarnings.cmake)

# Placeholder interface target so the skeleton configures cleanly.
# Concrete sources and public headers land in follow-up commits.
# Warning flags attach to real STATIC/OBJECT/EXECUTABLE targets later
# via parcae_set_project_warnings(<target>).
add_library(parcae_core INTERFACE)
add_library(parcae::core ALIAS parcae_core)
target_include_directories(parcae_core INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if(PARCAE_BUILD_TESTS)
  enable_testing()
  # tests/ CMakeLists.txt added when Catch2 wiring lands
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

if(PARCAE_BUILD_TOOLS)
  # tools/ CMakeLists.txt added when CLIs land
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/CMakeLists.txt")
    add_subdirectory(tools)
  endif()
endif()

message(STATUS "Parcae ${PROJECT_VERSION}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  PARCAE_BUILD_TESTS=${PARCAE_BUILD_TESTS}")
message(STATUS "  PARCAE_BUILD_TOOLS=${PARCAE_BUILD_TOOLS}")
