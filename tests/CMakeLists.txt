include(Catch)

add_executable(parcae_tests
  smoke.cpp
  z29_test.cpp
  token_stream_test.cpp
  gematria_profile_test.cpp
  corpus_test.cpp
)

target_link_libraries(parcae_tests PRIVATE
  parcae::core
  Catch2::Catch2WithMain
)

target_compile_definitions(parcae_tests PRIVATE
  PARCAE_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
)

parcae_set_project_warnings(parcae_tests)

catch_discover_tests(parcae_tests)
