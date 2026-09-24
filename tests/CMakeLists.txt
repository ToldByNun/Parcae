include(Catch)

find_program(PARCAE_PYTHON_EXECUTABLE NAMES python python3 py)
if(NOT PARCAE_PYTHON_EXECUTABLE)
  set(PARCAE_PYTHON_EXECUTABLE "python")
endif()

add_executable(parcae_tests
  smoke.cpp
  z29_test.cpp
  dsl_spec_version_test.cpp
  dsl_diag_test.cpp
  dsl_ast_ingest_test.cpp
  dsl_semantic_gate_test.cpp
  dsl_scope_analyzer_test.cpp
  dsl_divergence_gate_test.cpp
  dsl_directive_table_test.cpp
  dsl_smart_compiler_golden_test.cpp
  dsl_build_select_test.cpp
  dsl_host_glue_test.cpp
  dsl_z29_expr_test.cpp
  dsl_ir_test.cpp
  dsl_forbidden_ast_test.cpp
  dsl_ast_ingest_fuzz_test.cpp
  dsl_ir_applicator_test.cpp
  dsl_emit_cpu_test.cpp
  dsl_emit_cuda_test.cpp
  dsl_poly2_oracle_test.cpp
  dsl_verify_test.cpp
  dsl_fuse_test.cpp
  dsl_koan_parity_test.cpp
  dsl_optimize_test.cpp
  dsl_launch_plan_test.cpp
  dsl_peak_sanity_test.cpp
  dsl_theory_artifact_test.cpp
  dsl_theory_registry_test.cpp
  dsl_theory_validate_test.cpp
  dsl_theory_sweep_test.cpp
  dsl_theory_envelope_test.cpp
  dsl_theory_dispatch_test.cpp
  dsl_spec_major_bump_test.cpp
  dsl_compile_test.cpp
  dsl_examples_new_math_test.cpp
  dsl_examples_full_lifecycle_test.cpp
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
  rank_cuda_ordering_test.cpp
  agent_policy_test.cpp
  agent_policy_cli_test.cpp
  hypothesis_record_test.cpp
  hypothesis_sandbox_test.cpp
  search_job_test.cpp
  search_adversarial_test.cpp
  search_prior_test.cpp
  batch_artifact_test.cpp
  batch_artifact_limits_test.cpp
  hypothesis_bridge_test.cpp
  hypothesis_bridge_score_test.cpp
  search_roundtrip_test.cpp
  workspace_cipher_test.cpp
  workspace_cipher_resolve_test.cpp
  gpu_candidate_export_test.cpp
  cpu_candidate_export_test.cpp
  cpu_gpu_export_parity_test.cpp
  compose_export_parity_test.cpp
  search_scheduler_test.cpp
  search_cycle_cli_test.cpp
  parity_test.cpp
  transform_property_test.cpp
  cuda_property_test.cpp
  cuda_device_buffer_test.cpp
  cuda_params_test.cpp
  cuda_interrupt_device_view_test.cpp
  cuda_candidate_batch_buffers_test.cpp
  cuda_backend_test.cpp
  cuda_smoke_test.cpp
  cuda_dsl_test.cpp
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
  console_progress_mode_test.cpp
  console_ansi_test.cpp
  console_dashboard_test.cpp
)

target_link_libraries(parcae_tests PRIVATE
  parcae::core
  Catch2::Catch2WithMain
)

target_compile_definitions(parcae_tests PRIVATE
  PARCAE_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
  PARCAE_DSL_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/dsl"
  PARCAE_EXAMPLES_DIR="${CMAKE_SOURCE_DIR}/theories/examples"
  PARCAE_PYTHON_DIR="${CMAKE_SOURCE_DIR}/python"
  PARCAE_PYTHON_EXE="${PARCAE_PYTHON_EXECUTABLE}"
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
    parcae-search-cycle
    parcae-generate
    parcae-rank
    parcae-hypothesis
  )
  set(_parcae_cli_paths_dir "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
  file(GENERATE
    OUTPUT "${_parcae_cli_paths_dir}/parcae_cli_paths.h"
    CONTENT "#pragma once\n#define PARCAE_CLI_TOKENIZE R\"($<TARGET_FILE:parcae-tokenize>)\"\n#define PARCAE_CLI_DECODE R\"($<TARGET_FILE:parcae-decode>)\"\n#define PARCAE_CLI_SCORE R\"($<TARGET_FILE:parcae-score>)\"\n#define PARCAE_CLI_VALIDATE R\"($<TARGET_FILE:parcae-validate>)\"\n#define PARCAE_CLI_SEARCH_RUN R\"($<TARGET_FILE:parcae-search-run>)\"\n#define PARCAE_CLI_SEARCH_CYCLE R\"($<TARGET_FILE:parcae-search-cycle>)\"\n#define PARCAE_CLI_GENERATE R\"($<TARGET_FILE:parcae-generate>)\"\n#define PARCAE_CLI_RANK R\"($<TARGET_FILE:parcae-rank>)\"\n#define PARCAE_CLI_HYPOTHESIS R\"($<TARGET_FILE:parcae-hypothesis>)\"\n"
  )
  target_include_directories(parcae_tests PRIVATE "${_parcae_cli_paths_dir}")
  target_compile_definitions(parcae_tests PRIVATE PARCAE_HAS_CLI_GOLDENS=1)
endif()

parcae_set_project_warnings(parcae_tests)

# MSVC: allow parallel CL.EXE writers to the same .pdb (Debug).
if(MSVC)
  target_compile_options(parcae_tests PRIVATE /FS)
endif()

catch_discover_tests(parcae_tests)
