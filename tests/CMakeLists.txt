include(Catch)

add_executable(parcae_tests
  smoke.cpp
  z29_test.cpp
  token_stream_test.cpp
  gematria_profile_test.cpp
  corpus_test.cpp
  transform_test.cpp
  primes_test.cpp
  validation_test.cpp
  hash_test.cpp
  score_test.cpp
  generate_test.cpp
  batch_test.cpp
  tool_api_test.cpp
  tool_response_test.cpp
  hypothesis_record_test.cpp
  hypothesis_sandbox_test.cpp
  parity_test.cpp
  transform_property_test.cpp
  cuda_property_test.cpp
  cuda_device_buffer_test.cpp
  cuda_params_test.cpp
  cuda_interrupt_device_view_test.cpp
  cuda_candidate_batch_buffers_test.cpp
  cuda_backend_test.cpp
  cuda_smoke_test.cpp
  cuda_atbash_test.cpp
  cuda_atbash_batch_test.cpp
  cuda_caesar_test.cpp
  cuda_caesar_batch_test.cpp
  cuda_identity_test.cpp
  cuda_affine_test.cpp
  cuda_affine_batch_test.cpp
  cuda_vigenere_test.cpp
  cuda_vigenere_batch_test.cpp
  cuda_beaufort_test.cpp
  cuda_totient_test.cpp
  cuda_compose_test.cpp
  cuda_golden_parity_test.cpp
  cuda_solved_parity_test.cpp
  cuda_score_exact_hamming_test.cpp
  cuda_score_ic_mod29_test.cpp
  cuda_score_self_repeat_test.cpp
  cuda_score_chi2_test.cpp
  cuda_score_dispatch_test.cpp
  cuda_score_noise_test.cpp
  cuda_batch_score_test.cpp
  cuda_batch_rank1_test.cpp
  cuda_caesar_chi2_batch_test.cpp
  search_run_test.cpp
)

target_link_libraries(parcae_tests PRIVATE
  parcae::core
  Catch2::Catch2WithMain
)

target_compile_definitions(parcae_tests PRIVATE
  PARCAE_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
  PARCAE_GOLDENS_TOOL_JSON_DIR="${CMAKE_SOURCE_DIR}/tests/goldens/tool-json"
)

# Host-side CUDA façade headers (params POD / JSON) live under Parcae/Parcae/cuda/.
target_include_directories(parcae_tests PRIVATE
  "${CMAKE_SOURCE_DIR}/Parcae/Parcae/cuda"
)

if(TARGET parcae::cuda)
  target_link_libraries(parcae_tests PRIVATE parcae::cuda)
endif()

# CLI golden tests: emit a per-config header so Windows paths are not
# mangled by C string escapes (\t in \tools, etc.).
if(TARGET parcae-tokenize)
  target_sources(parcae_tests PRIVATE tool_json_golden_test.cpp)
  add_dependencies(parcae_tests
    parcae-tokenize
    parcae-decode
    parcae-score
    parcae-validate
    parcae-search-run
    parcae-generate
    parcae-rank
    parcae-hypothesis
  )
  set(_parcae_cli_paths_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
  file(GENERATE
    OUTPUT "${_parcae_cli_paths_dir}/parcae_cli_paths.h"
    CONTENT "#pragma once\n#define PARCAE_CLI_TOKENIZE R\"($<TARGET_FILE:parcae-tokenize>)\"\n#define PARCAE_CLI_DECODE R\"($<TARGET_FILE:parcae-decode>)\"\n#define PARCAE_CLI_SCORE R\"($<TARGET_FILE:parcae-score>)\"\n#define PARCAE_CLI_VALIDATE R\"($<TARGET_FILE:parcae-validate>)\"\n#define PARCAE_CLI_SEARCH_RUN R\"($<TARGET_FILE:parcae-search-run>)\"\n#define PARCAE_CLI_GENERATE R\"($<TARGET_FILE:parcae-generate>)\"\n#define PARCAE_CLI_RANK R\"($<TARGET_FILE:parcae-rank>)\"\n#define PARCAE_CLI_HYPOTHESIS R\"($<TARGET_FILE:parcae-hypothesis>)\"\n"
  )
  target_include_directories(parcae_tests PRIVATE "${_parcae_cli_paths_dir}")
  target_compile_definitions(parcae_tests PRIVATE PARCAE_HAS_CLI_GOLDENS=1)
endif()

parcae_set_project_warnings(parcae_tests)

catch_discover_tests(parcae_tests)
