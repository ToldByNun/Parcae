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
  # Optional CUDA twins for --backend cuda (same sources as parcae_tests).
  if(TARGET parcae::cuda)
    target_link_libraries(${name} PRIVATE parcae::cuda)
    target_include_directories(${name} PRIVATE
      "${CMAKE_SOURCE_DIR}/Parcae/Parcae/cuda"
    )
  endif()
  parcae_set_project_warnings(${name})
endfunction()

parcae_add_cli(parcae-tokenize parcae_tokenize/main.cpp)
parcae_add_cli(parcae-decode parcae_decode/main.cpp)
parcae_add_cli(parcae-score parcae_score/main.cpp)
parcae_add_cli(parcae-validate parcae_validate/main.cpp)
parcae_add_cli(parcae-parity-gen parcae_parity_gen/main.cpp)
parcae_add_cli(parcae-parity parcae_parity/main.cpp)
parcae_add_cli(parcae-search-run parcae_search_run/main.cpp)
parcae_add_cli(parcae-blind-crack parcae_blind_crack/main.cpp)
parcae_add_cli(parcae-throughput-tiers parcae_throughput_tiers/main.cpp)

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

  if(TARGET parcae::cuda)
    add_test(
      NAME cli_decode_flags_atbash_cuda
      COMMAND $<TARGET_FILE:parcae-decode>
        --data-dir "${CMAKE_SOURCE_DIR}/data"
        --input "${CMAKE_SOURCE_DIR}/data/fixtures/solved/a-warning/ciphertext.txt"
        --transform-id atbash
        --direction decrypt
        --backend cuda
    )
    add_test(
      NAME cli_score_ic_indices_cuda
      COMMAND $<TARGET_FILE:parcae-score>
        --data-dir "${CMAKE_SOURCE_DIR}/data"
        --score-id ic_mod29
        --indices
        --backend cuda
        --input "${CMAKE_SOURCE_DIR}/data/fixtures/cli/score-indices.txt"
    )
  endif()

  add_test(
    NAME cli_score_ic_indices
    COMMAND $<TARGET_FILE:parcae-score>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --score-id ic_mod29
      --indices
      --input "${CMAKE_SOURCE_DIR}/data/fixtures/cli/score-indices.txt"
  )

  add_test(
    NAME cli_score_list_json
    COMMAND $<TARGET_FILE:parcae-score>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --list
      --json
  )

  add_test(
    NAME cli_parity_check_all
    COMMAND $<TARGET_FILE:parcae-parity>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      check
      --all
      --json
  )

  if(TARGET parcae::cuda)
    add_test(
      NAME cli_parity_check_all_cuda
      COMMAND $<TARGET_FILE:parcae-parity>
        --data-dir "${CMAKE_SOURCE_DIR}/data"
        check
        --all
        --compare-cuda
        --json
    )
    add_test(
      NAME cli_parity_dump_caesar_cuda
      COMMAND $<TARGET_FILE:parcae-parity>
        --data-dir "${CMAKE_SOURCE_DIR}/data"
        dump
        --name caesar-s3-len64
        --backend cuda
    )
    add_test(
      NAME cli_search_run_caesar_cuda
      COMMAND $<TARGET_FILE:parcae-search-run>
        --data-dir "${CMAKE_SOURCE_DIR}/data"
        --backend cuda
        --seed 2109016688
        --stream-length 256
        --repeats 2
    )
  endif()

  add_test(
    NAME cli_search_run_caesar_cpu
    COMMAND $<TARGET_FILE:parcae-search-run>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --backend cpu
      --seed 2109016688
      --stream-length 256
      --repeats 2
  )

  add_test(
    NAME cli_validate_id_a_warning
    COMMAND $<TARGET_FILE:parcae-validate>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --id a-warning
      --require-locked
  )

  add_test(
    NAME cli_validate_all_locked
    COMMAND $<TARGET_FILE:parcae-validate>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --all
      --require-locked
      --json
  )
endif()
