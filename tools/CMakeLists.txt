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

find_program(PARCAE_PYTHON_EXECUTABLE NAMES python python3 py)
if(NOT PARCAE_PYTHON_EXECUTABLE)
  set(PARCAE_PYTHON_EXECUTABLE "python")
endif()

parcae_add_cli(parcae-tokenize parcae_tokenize/main.cpp)
parcae_add_cli(parcae-decode parcae_decode/main.cpp)
parcae_add_cli(parcae-score parcae_score/main.cpp)
parcae_add_cli(parcae-validate parcae_validate/main.cpp)
parcae_add_cli(parcae-catalog parcae_catalog/main.cpp)
parcae_add_cli(parcae-compile parcae_compile/main.cpp)
target_compile_definitions(parcae-compile PRIVATE
  PARCAE_PYTHON_DIR="${CMAKE_SOURCE_DIR}/python"
  PARCAE_PYTHON_EXE="${PARCAE_PYTHON_EXECUTABLE}"
)
parcae_add_cli(parcae-generate parcae_generate/main.cpp)
parcae_add_cli(parcae-rank parcae_rank/main.cpp)
parcae_add_cli(parcae-hypothesis parcae_hypothesis/main.cpp)
parcae_add_cli(parcae-parity-gen parcae_parity_gen/main.cpp)
parcae_add_cli(parcae-parity parcae_parity/main.cpp)
parcae_add_cli(parcae-search-run parcae_search_run/main.cpp)
parcae_add_cli(parcae-blind-crack parcae_blind_crack/main.cpp)
parcae_add_cli(parcae-throughput-tiers parcae_throughput_tiers/main.cpp)

if(PARCAE_BUILD_TESTS)
  add_test(
    NAME cli_catalog_json
    COMMAND $<TARGET_FILE:parcae-catalog>
      --json
  )

  add_test(
    NAME cli_compile_status_json
    COMMAND $<TARGET_FILE:parcae-compile>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --status
      --json
  )

  add_test(
    NAME cli_compile_quadratic_polynomial_stream
    COMMAND $<TARGET_FILE:parcae-compile>
      --data-dir "${CMAKE_BINARY_DIR}/dsl_compile_testdata"
      --json
      "${CMAKE_SOURCE_DIR}/tests/fixtures/dsl/sources/quadratic_polynomial_stream.py"
  )

  add_test(
    NAME cli_compile_missing_path_json
    COMMAND $<TARGET_FILE:parcae-compile>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --json
  )
  set_tests_properties(cli_compile_missing_path_json PROPERTIES WILL_FAIL TRUE)

  add_test(
    NAME cli_validate_theories_empty_json
    COMMAND $<TARGET_FILE:parcae-validate>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --theories
      --json
  )

  add_test(
    NAME cli_catalog_generators_json
    COMMAND $<TARGET_FILE:parcae-catalog>
      --generators
      --json
  )

  add_test(
    NAME cli_generate_caesar_indices_json
    COMMAND $<TARGET_FILE:parcae-generate>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --generator-id gen_caesar
      --indices
      --input "${CMAKE_SOURCE_DIR}/data/fixtures/cli/score-indices.txt"
      --json
  )

  add_test(
    NAME cli_generate_list_json
    COMMAND $<TARGET_FILE:parcae-generate>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --list
      --json
  )

  add_test(
    NAME cli_rank_caesar_ic_json
    COMMAND $<TARGET_FILE:parcae-rank>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --candidates "${CMAKE_SOURCE_DIR}/data/fixtures/cli/rank-candidates.json"
      --score-id ic_mod29
      --k 3
      --json
  )

  add_test(
    NAME cli_hypothesis_list_example_json
    COMMAND $<TARGET_FILE:parcae-hypothesis>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      list
      --workspace _example
      --json
  )

  add_test(
    NAME cli_hypothesis_show_example_json
    COMMAND $<TARGET_FILE:parcae-hypothesis>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      show
      --workspace _example
      --id h-atbash-example
      --json
  )

  add_test(
    NAME cli_hypothesis_init_data_dir_under_fixtures_denied
    COMMAND $<TARGET_FILE:parcae-hypothesis>
      --data-dir "${CMAKE_SOURCE_DIR}/data/fixtures/solved/a-warning"
      init
      --workspace evil-ws
      --id h-evil
      --json
  )
  set_tests_properties(
    cli_hypothesis_init_data_dir_under_fixtures_denied PROPERTIES WILL_FAIL TRUE)

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

  add_test(
    NAME cli_decode_rebuild_text_json
    COMMAND $<TARGET_FILE:parcae-decode>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --input "${CMAKE_SOURCE_DIR}/data/fixtures/solved/a-warning/ciphertext.txt"
      --transform-id atbash
      --direction decrypt
      --rebuild-text
      --json
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
        --allow-cuda
    )
    add_test(
      NAME cli_score_ic_indices_cuda
      COMMAND $<TARGET_FILE:parcae-score>
        --data-dir "${CMAKE_SOURCE_DIR}/data"
        --score-id ic_mod29
        --indices
        --backend cuda
        --allow-cuda
        --input "${CMAKE_SOURCE_DIR}/data/fixtures/cli/score-indices.txt"
    )
  endif()

  add_test(
    NAME cli_score_cuda_requires_allow_cuda
    COMMAND $<TARGET_FILE:parcae-score>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --score-id ic_mod29
      --indices
      --backend cuda
      --input "${CMAKE_SOURCE_DIR}/data/fixtures/cli/score-indices.txt"
      --json
  )
  set_tests_properties(cli_score_cuda_requires_allow_cuda PROPERTIES WILL_FAIL TRUE)

  add_test(
    NAME cli_score_ic_indices
    COMMAND $<TARGET_FILE:parcae-score>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --score-id ic_mod29
      --indices
      --input "${CMAKE_SOURCE_DIR}/data/fixtures/cli/score-indices.txt"
  )

  add_test(
    NAME cli_tokenize_json_usage_envelope
    COMMAND $<TARGET_FILE:parcae-tokenize>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --json
  )
  set_tests_properties(cli_tokenize_json_usage_envelope PROPERTIES WILL_FAIL TRUE)

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
    NAME cli_search_run_caesar_cpu_omit_timing_json
    COMMAND $<TARGET_FILE:parcae-search-run>
      --data-dir "${CMAKE_SOURCE_DIR}/data"
      --backend cpu
      --seed 2109016688
      --stream-length 256
      --repeats 2
      --json
      --omit-timing
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
