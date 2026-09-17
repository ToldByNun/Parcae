add_library(parcae_cli_common INTERFACE)
target_include_directories(parcae_cli_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(parcae_cli_common INTERFACE parcae::core)

function(parcae_add_cli name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE parcae::core parcae_cli_common)
  target_compile_definitions(${name} PRIVATE
    PARCAE_DEFAULT_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
  )
  parcae_set_project_warnings(${name})
endfunction()

parcae_add_cli(parcae-tokenize parcae_tokenize/main.cpp)
parcae_add_cli(parcae-decode parcae_decode/main.cpp)

if(PARCAE_BUILD_TESTS)
  add_test(
    NAME cli_tokenize_a_warning
    COMMAND $<TARGET_FILE:parcae-tokenize>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --json
      "${CMAKE_SOURCE_DIR}/data/fixtures/solved/a-warning/ciphertext.txt"
  )

  add_test(
    NAME cli_decode_manifest_a_warning
    COMMAND $<TARGET_FILE:parcae-decode>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --manifest "${CMAKE_SOURCE_DIR}/data/fixtures/solved/a-warning"
  )

  add_test(
    NAME cli_decode_flags_atbash
    COMMAND $<TARGET_FILE:parcae-decode>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --input "${CMAKE_SOURCE_DIR}/data/fixtures/solved/a-warning/ciphertext.txt"
      --transform-id atbash
      --direction decrypt
  )
endif()
