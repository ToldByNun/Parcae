# Pinned third-party dependencies (FetchContent).
# Include from the root CMakeLists.txt after project() is defined.

include(FetchContent)

set(PARCAE_CATCH2_VERSION "v3.7.1" CACHE STRING "Pinned Catch2 git tag")
set(PARCAE_NLOHMANN_JSON_VERSION "v3.11.3" CACHE STRING "Pinned nlohmann/json git tag")

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        ${PARCAE_NLOHMANN_JSON_VERSION}
  GIT_SHALLOW    TRUE
)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json)

if(PARCAE_BUILD_TESTS)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        ${PARCAE_CATCH2_VERSION}
    GIT_SHALLOW    TRUE
  )

  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(Catch2)

  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()
