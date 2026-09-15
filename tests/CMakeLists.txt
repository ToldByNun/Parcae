include(Catch)

add_executable(parcae_tests
  smoke.cpp
  z29_test.cpp
  token_stream_test.cpp
)

target_link_libraries(parcae_tests PRIVATE
  parcae::core
  Catch2::Catch2WithMain
)

parcae_set_project_warnings(parcae_tests)

catch_discover_tests(parcae_tests)
