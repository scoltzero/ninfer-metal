cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(OMLX_M5_BUILD_PROBES "Build Metal capability and bandwidth probes" ON)
option(OMLX_M5_BUILD_PYTHON "Build the nanobind smoke extension" ON)
option(OMLX_M5_BUILD_NAX_METALLIB "Cross-compile pure Metal 4 / NAX kernel libraries" ON)
option(OMLX_M5_BUILD_MLX_LEGACY_SMOKE "Build legacy MLX-header smoke kernels" OFF)
option(OMLX_M5_BUILD_MLX_FIXTURES "Build Python MLX/dflash-mlx oracle fixtures and dependent tests" ON)

set(
  NINFER_METAL_DISPLAY_VERSION
  "miv-v0.03"
  CACHE STRING
  "Display version string reported by mseries_backendd")
set(
  NINFER_METAL_CODENAME
  "miv"
  CACHE STRING
  "Codename reported by mseries_backendd")
set(
  NINFER_METAL_SEMVER
  "0.0.3"
  CACHE STRING
  "Semantic version reported by mseries_backendd")
set(
  NINFER_METAL_CORE_ABI
  "1"
  CACHE STRING
  "Core ABI version integer reported by mseries_backendd")

set(
  OMLX_M5_DEVELOPER_DIR
  "/Applications/Xcode.app/Contents/Developer"
  CACHE PATH
  "Xcode developer directory used for xcrun/Metal compilation")

set(_omlx_default_third_party_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
if(NOT EXISTS "${_omlx_default_third_party_dir}/nlohmann/json.hpp")
  if(EXISTS "${CMAKE_SOURCE_DIR}/ninfer/third_party/nlohmann/json.hpp")
    set(_omlx_default_third_party_dir "${CMAKE_SOURCE_DIR}/ninfer/third_party")
  endif()
endif()
if(NOT EXISTS "${_omlx_default_third_party_dir}/nlohmann/json.hpp")
  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --path-format=absolute --git-common-dir
    OUTPUT_VARIABLE _omlx_git_common_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _omlx_git_common_res)
  if(_omlx_git_common_res EQUAL 0 AND _omlx_git_common_dir)
    cmake_path(GET _omlx_git_common_dir PARENT_PATH _omlx_main_repo_dir)
    if(EXISTS "${_omlx_main_repo_dir}/ninfer/third_party/nlohmann/json.hpp")
      set(_omlx_default_third_party_dir "${_omlx_main_repo_dir}/ninfer/third_party")
    endif()
  endif()
endif()
if(NOT EXISTS "${_omlx_default_third_party_dir}/nlohmann/json.hpp")
  foreach(_candidate_inc IN ITEMS "/opt/homebrew/include" "/usr/local/include")
    if(EXISTS "${_candidate_inc}/nlohmann/json.hpp")
      set(_omlx_default_third_party_dir "${_candidate_inc}")
      break()
    endif()
  endforeach()
endif()
set(
  OMLX_M5_THIRD_PARTY_INCLUDE_DIR
  "${_omlx_default_third_party_dir}"
  CACHE PATH
  "Third-party C++ header directory providing nlohmann/json.hpp")

add_library(
  omlx_m5_core_runtime
  STATIC
  src/activation.cpp
  src/artifact.cpp
  src/bounded_batch.cpp
  src/capability.cpp
  src/dflash2.cpp
  src/dflash2_program.cpp
  src/dflash_forward.cpp
  src/dispatch.cpp
  src/engine.cpp
  src/execution_contract.cpp
  src/fake_backend.cpp
  src/hw_counters.mm
  src/kv_page_pool.cpp
  src/layer_schedule.cpp
  src/lora.cpp
  src/materialization.cpp
  src/metal.mm
  src/model_config.cpp
  src/mtp_forward.cpp
  src/native_backend.cpp
  src/position_sampling.cpp
  src/quantization.cpp
  src/qwen_forward.cpp
  src/qwen_memory.cpp
  src/qwen_program.cpp
  src/safetensors.cpp
  src/state.cpp
  src/target_verify_gpu.cpp
  src/topology_profile.cpp)
target_include_directories(
  omlx_m5_core_runtime
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_definitions(
  omlx_m5_core_runtime
  PUBLIC
  NINFER_METAL_DISPLAY_VERSION="${NINFER_METAL_DISPLAY_VERSION}"
  NINFER_METAL_CODENAME="${NINFER_METAL_CODENAME}"
  NINFER_METAL_SEMVER="${NINFER_METAL_SEMVER}"
  NINFER_METAL_CORE_ABI=${NINFER_METAL_CORE_ABI})
target_compile_features(omlx_m5_core_runtime PUBLIC cxx_std_20)
target_compile_options(
  omlx_m5_core_runtime
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)
target_compile_options(
  omlx_m5_core_runtime
  PRIVATE
  "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  omlx_m5_core_runtime
  PUBLIC
  "-framework IOSurface"
  "-framework CoreFoundation"
  "-framework Foundation"
  "-framework IOKit"
  "-framework Metal")

add_library(
  omlx_m5_probe_harnesses
  STATIC
  src/affine_lane.cpp
  src/attention.cpp
  src/gdn.cpp
  src/memory_plan.cpp
  src/mlp.cpp
  src/moe.cpp
  src/mxfp4_small_t.cpp
  src/normalization.cpp
  src/p11_tensor_view.cpp
  src/program.cpp
  src/projection.cpp
  src/qmm_broad.cpp
  src/qmm_small_t.cpp
  src/qmv.cpp
  src/reference.cpp
  src/vocab.cpp)
target_include_directories(
  omlx_m5_probe_harnesses
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_features(omlx_m5_probe_harnesses PUBLIC cxx_std_20)
target_compile_options(
  omlx_m5_probe_harnesses
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)
target_link_libraries(
  omlx_m5_probe_harnesses
  PUBLIC
  omlx_m5_core_runtime)

add_library(omlx_m5_runtime INTERFACE)
target_link_libraries(
  omlx_m5_runtime
  INTERFACE
  omlx_m5_core_runtime
  omlx_m5_probe_harnesses)

add_executable(
  omlx_m5_native_contract_tests
  tests/native_contract_tests.cpp)
target_link_libraries(
  omlx_m5_native_contract_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_native_contract
  COMMAND omlx_m5_native_contract_tests)

add_executable(
  omlx_m5_b0_artifact_tests
  tests/b0_artifact_tests.cpp)
target_include_directories(
  omlx_m5_b0_artifact_tests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
  omlx_m5_b0_artifact_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_b0_artifact
  COMMAND omlx_m5_b0_artifact_tests)

add_executable(
  omlx_m5_b1_materialization_tests
  tests/b1_materialization_tests.cpp)
target_link_libraries(
  omlx_m5_b1_materialization_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_b1_materialization
  COMMAND omlx_m5_b1_materialization_tests)

add_executable(
  omlx_m5_b1_activation_tests
  tests/b1_activation_tests.cpp)
target_link_libraries(
  omlx_m5_b1_activation_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_b1_activation
  COMMAND omlx_m5_b1_activation_tests)

add_executable(
  omlx_m5_b2_dispatch_tests
  tests/b2_dispatch_tests.cpp)
target_link_libraries(
  omlx_m5_b2_dispatch_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_b2_dispatch
  COMMAND omlx_m5_b2_dispatch_tests)

add_executable(
  omlx_m5_inspect_artifact
  apps/inspect_artifact.cpp)
target_include_directories(
  omlx_m5_inspect_artifact
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_artifact
  PRIVATE
  omlx_m5_runtime)

add_executable(
  omlx_m5_inspect_materialization
  apps/inspect_materialization.cpp)
target_include_directories(
  omlx_m5_inspect_materialization
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_materialization
  PRIVATE
  omlx_m5_runtime)

add_executable(
  omlx_m5_inspect_activation
  apps/inspect_activation.cpp)
target_include_directories(
  omlx_m5_inspect_activation
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_activation
  PRIVATE
  omlx_m5_runtime)

add_executable(
  omlx_m5_inspect_dispatch
  apps/inspect_dispatch.cpp)
target_include_directories(
  omlx_m5_inspect_dispatch
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_dispatch
  PRIVATE
  omlx_m5_runtime)

add_executable(
  omlx_m5_d1_host_tests
  tests/d1_host_tests.cpp)
target_include_directories(
  omlx_m5_d1_host_tests
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_d1_host_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_d1_host
  COMMAND omlx_m5_d1_host_tests)

add_executable(
  omlx_m5_d3_state_spec_tests
  tests/d3_state_spec_tests.cpp)
target_link_libraries(
  omlx_m5_d3_state_spec_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_d3_state_spec
  COMMAND omlx_m5_d3_state_spec_tests)

add_executable(
  omlx_m5_d2_reference_tests
  tests/d2_reference_tests.cpp)
target_link_libraries(
  omlx_m5_d2_reference_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_d2_reference
  COMMAND omlx_m5_d2_reference_tests)

add_executable(
  omlx_m5_reference_runner
  apps/reference_runner.cpp)
target_include_directories(
  omlx_m5_reference_runner
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_reference_runner
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_d2_runner_256
  COMMAND omlx_m5_reference_runner --decode-tokens 256 --chunk-size 13)

add_executable(
  omlx_m5_reference_projection
  apps/reference_projection.cpp)
target_include_directories(
  omlx_m5_reference_projection
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_reference_projection
  PRIVATE
  omlx_m5_runtime)

add_executable(
  omlx_m5_reference_layer
  apps/reference_layer.cpp)
target_include_directories(
  omlx_m5_reference_layer
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_reference_layer
  PRIVATE
  omlx_m5_runtime)

add_executable(
  omlx_m5_inspect_model
  apps/inspect_model.cpp)
target_include_directories(
  omlx_m5_inspect_model
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_model
  PRIVATE
  omlx_m5_runtime)

set(
  OMLX_M5_MODEL_DIR
  "${CMAKE_SOURCE_DIR}/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
  CACHE PATH
  "Fixed Qwen3.8-27B oQ4e checkpoint used by D1 integration tests")
if(EXISTS "${OMLX_M5_MODEL_DIR}/model.safetensors.index.json")
  add_executable(
    omlx_m5_d1_model_integration_tests
    tests/d1_model_integration_tests.cpp)
  target_link_libraries(
    omlx_m5_d1_model_integration_tests
    PRIVATE
    omlx_m5_runtime)
  add_test(
    NAME omlx_m5_d1_model_integration
    COMMAND
      omlx_m5_d1_model_integration_tests
      "${OMLX_M5_MODEL_DIR}")
else()
  message(
    STATUS
    "D1 target checkpoint not found at ${OMLX_M5_MODEL_DIR}; real-model tests are unavailable")
endif()

set(
  OMLX_M5_SMALL_MODEL_0_8B_DIR
  "${CMAKE_SOURCE_DIR}/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
  CACHE PATH
  "Pinned Qwen3.5-0.8B MLX fixture used by D2 end-to-end tests")
set(
  OMLX_M5_SMALL_MODEL_4B_DIR
  "${CMAKE_SOURCE_DIR}/models/mlx-community/Qwen3.5-4B-MLX-4bit"
  CACHE PATH
  "Pinned Qwen3.5-4B MLX fixture used by D2 end-to-end tests")

find_program(
  OMLX_M5_XCRUN
  xcrun
  HINTS "${OMLX_M5_DEVELOPER_DIR}/usr/bin"
  REQUIRED)

set(OMLX_M5_METAL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metallib")
file(MAKE_DIRECTORY "${OMLX_M5_METAL_OUTPUT_DIR}")

function(omlx_m5_compile_metallib target source output_name deployment_target)
  set(options FAST_MATH)
  set(one_value_args)
  set(multi_value_args INCLUDE_DIRS EXTRA_FLAGS EXTRA_DEPENDS)
  cmake_parse_arguments(
    ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN})

  get_filename_component(source_name "${source}" NAME_WE)
  set(air_file "${OMLX_M5_METAL_OUTPUT_DIR}/${source_name}.air")
  set(metallib_file "${OMLX_M5_METAL_OUTPUT_DIR}/${output_name}")
  set(include_args)
  foreach(include_dir IN LISTS ARG_INCLUDE_DIRS)
    list(APPEND include_args "-I${include_dir}")
  endforeach()
  if(ARG_FAST_MATH)
    set(_omlx_m5_math_flag "-ffast-math")
  else()
    set(_omlx_m5_math_flag "-fno-fast-math")
  endif()

  add_custom_command(
    OUTPUT "${air_file}"
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "DEVELOPER_DIR=${OMLX_M5_DEVELOPER_DIR}"
      "${OMLX_M5_XCRUN}" -sdk macosx metal
      -x metal
      -O3
      -Wall
      -Wextra
      "${_omlx_m5_math_flag}"
      "-mmacosx-version-min=${deployment_target}"
      ${ARG_EXTRA_FLAGS}
      ${include_args}
      -c "${source}"
      -o "${air_file}"
    DEPENDS "${source}" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fp4_codec.h" ${ARG_EXTRA_DEPENDS}
    COMMENT "Compiling ${source_name}.metal"
    VERBATIM)

  add_custom_command(
    OUTPUT "${metallib_file}"
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "DEVELOPER_DIR=${OMLX_M5_DEVELOPER_DIR}"
      "${OMLX_M5_XCRUN}" -sdk macosx metal
      "-mmacosx-version-min=${deployment_target}"
      "${air_file}"
      -o "${metallib_file}"
    DEPENDS "${air_file}"
    COMMENT "Linking ${output_name}"
    VERBATIM)

  add_custom_target("${target}" ALL DEPENDS "${metallib_file}")
  set("${target}_PATH" "${metallib_file}" PARENT_SCOPE)
endfunction()

omlx_m5_compile_metallib(
  omlx_m5_dflash_forward_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/dflash_forward.metal"
  "omlx_m5_dflash_forward.metallib"
  "15.0"
  EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/position_sampling_shared.h")

omlx_m5_compile_metallib(
  omlx_m5_target_verify_gpu_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/target_verify_gpu.metal"
  "omlx_m5_target_verify_gpu.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_attention_contiguous_control_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_contiguous_control.metal"
  "omlx_m5_attention_contiguous_control.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_attention_exact_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_exact.metal"
  "omlx_m5_attention_exact.metallib"
  "15.0"
  EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_postprocess.metal"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_split_kv.metal")

omlx_m5_compile_metallib(
  omlx_m5_qwen_forward_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qwen_forward.metal"
  "omlx_m5_qwen_forward.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_classic_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/smoke.metal"
  "omlx_m5_smoke.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_icb_canary_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/icb_canary.metal"
  "omlx_m5_icb_canary.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_qwen_ar_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qwen_ar.metal"
  "omlx_m5_qwen_ar.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_dflash2_program_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/dflash2_program.metal"
  "omlx_m5_dflash2_program.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_d3_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_replay.metal"
  "omlx_m5_d3.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_d2_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/reference_qmm.metal"
  "omlx_m5_d2.metallib"
  "15.0")

omlx_m5_compile_metallib(
  omlx_m5_qmv_t1_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qmv_t1.metal"
  "omlx_m5_qmv_t1.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_qmm_small_t_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qmm_small_t.metal"
  "omlx_m5_qmm_small_t.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_position_sampling_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/position_sampling.metal"
  "omlx_m5_position_sampling.metallib"
  "15.0"
  EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/position_sampling_shared.h")

omlx_m5_compile_metallib(
  omlx_m5_mxfp4_small_t_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mxfp4_small_t.metal"
  "omlx_m5_mxfp4_small_t.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_qmm_broad_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qmm_broad.metal"
  "omlx_m5_qmm_broad.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_quantized_matrix_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantized_matrix.metal"
  "omlx_m5_quantized_matrix.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_moe_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/moe.metal"
  "omlx_m5_moe.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_rmsnorm_projection_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rmsnorm_projection.metal"
  "omlx_m5_rmsnorm_projection.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_attention_postprocess_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_postprocess.metal"
  "omlx_m5_attention_postprocess.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_attention_split_kv_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_split_kv.metal"
  "omlx_m5_attention_split_kv.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_mlp_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mlp.metal"
  "omlx_m5_mlp.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_gdn_candidate_io_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_candidate_io.metal"
  "omlx_m5_gdn_candidate_io.metallib"
  "15.0"
  EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_candidate_io.h")

omlx_m5_compile_metallib(
  omlx_m5_gdn_prefill_candidates_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_prefill_candidates.metal"
  "omlx_m5_gdn_prefill_candidates.metallib"
  "15.0"
  EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_candidate_io.h")

omlx_m5_compile_metallib(
  omlx_m5_gdn_layer_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_layer.metal"
  "omlx_m5_gdn_layer.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

omlx_m5_compile_metallib(
  omlx_m5_vocab_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vocab.metal"
  "omlx_m5_vocab.metallib"
  "15.0"
  EXTRA_FLAGS
    -Wno-c++17-extensions)

add_executable(
  omlx_m5_b2_qmv_t1_tests
  tests/b2_qmv_t1_tests.cpp)
target_compile_definitions(
  omlx_m5_b2_qmv_t1_tests
  PRIVATE
  OMLX_M5_QMV_T1_METALLIB_PATH="${omlx_m5_qmv_t1_metallib_PATH}")
target_link_libraries(
  omlx_m5_b2_qmv_t1_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b2_qmv_t1_tests
  omlx_m5_qmv_t1_metallib)
add_test(
  NAME omlx_m5_b2_qmv_t1
  COMMAND omlx_m5_b2_qmv_t1_tests)

add_executable(
  omlx_m5_b2_qmm_small_t_tests
  tests/b2_qmm_small_t_tests.cpp)
target_compile_definitions(
  omlx_m5_b2_qmm_small_t_tests
  PRIVATE
  OMLX_M5_QMM_SMALL_T_METALLIB_PATH="${omlx_m5_qmm_small_t_metallib_PATH}")
target_link_libraries(
  omlx_m5_b2_qmm_small_t_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b2_qmm_small_t_tests
  omlx_m5_qmm_small_t_metallib)
add_test(
  NAME omlx_m5_b2_qmm_small_t
  COMMAND omlx_m5_b2_qmm_small_t_tests)

add_executable(
  omlx_m5_inspect_qmv_t1
  apps/inspect_qmv_t1.cpp)
target_include_directories(
  omlx_m5_inspect_qmv_t1
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_qmv_t1
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_qmv_t1
  omlx_m5_qmv_t1_metallib)

add_executable(
  omlx_m5_inspect_qmm_small_t
  apps/inspect_qmm_small_t.cpp)
target_include_directories(
  omlx_m5_inspect_qmm_small_t
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_qmm_small_t
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_qmm_small_t
  omlx_m5_qmm_small_t_metallib)

set(omlx_m5_nax_metallib_PATH "")
if(OMLX_M5_BUILD_NAX_METALLIB)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "DEVELOPER_DIR=${OMLX_M5_DEVELOPER_DIR}"
      "${OMLX_M5_XCRUN}" -sdk macosx --show-sdk-version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE OMLX_M5_MACOS_SDK_VERSION
    RESULT_VARIABLE OMLX_M5_SDK_QUERY_RESULT)
  if(NOT OMLX_M5_SDK_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Unable to query the macOS SDK version")
  endif()
  if(OMLX_M5_MACOS_SDK_VERSION VERSION_LESS 26.2)
    message(
      FATAL_ERROR
      "Metal 4/NAX compilation requires macOS SDK 26.2 or newer; found ${OMLX_M5_MACOS_SDK_VERSION}")
  endif()

  omlx_m5_compile_metallib(
    omlx_m5_affine_lane_metallib
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/affine_lane.metal"
    "omlx_m5_affine_lane.metallib"
    "26.0"
    EXTRA_FLAGS -std=metal4.0)

  if(OMLX_M5_MACOS_SDK_VERSION VERSION_GREATER_EQUAL 27.0)
    omlx_m5_compile_metallib(
      omlx_m5_p11_fp4_native_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/p11_fp4_native.metal"
      "omlx_m5_p11_fp4_native.metallib"
      "27.0")
    omlx_m5_compile_metallib(
      omlx_m5_mxfp4_nax_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mxfp4_nax.metal"
      "omlx_m5_mxfp4_nax.metallib"
      "27.0"
      EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    omlx_m5_compile_metallib(
      omlx_m5_affine_nax_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/affine_nax.metal"
      "omlx_m5_affine_nax.metallib"
      "27.0"
      EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    omlx_m5_compile_metallib(
      omlx_m5_bf16_nax_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/bf16_nax.metal"
      "omlx_m5_bf16_nax.metallib"
      "27.0"
      EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    omlx_m5_compile_metallib(
      omlx_m5_moe_nax_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/moe_nax.metal"
      "omlx_m5_moe_nax.metallib"
      "27.0"
      EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_nax_prefill.metal")
      omlx_m5_compile_metallib(
        omlx_m5_attention_nax_prefill_metallib
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_nax_prefill.metal"
        "omlx_m5_attention_nax_prefill.metallib"
        "27.0"
        FAST_MATH
        EXTRA_FLAGS -std=metal4.0
        EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_affine4.metal")
      omlx_m5_compile_metallib(
        omlx_m5_attention_affine4_metallib
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_affine4.metal"
        "omlx_m5_attention_affine4.metallib"
        "27.0"
        FAST_MATH
        EXTRA_FLAGS -std=metal4.0
        EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_sage2.metal")
      omlx_m5_compile_metallib(
        omlx_m5_attention_sage2_metallib
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_sage2.metal"
        "omlx_m5_attention_sage2.metallib"
        "27.0"
        FAST_MATH
        EXTRA_FLAGS -std=metal4.1)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_minference.metal")
      omlx_m5_compile_metallib(
        omlx_m5_attention_minference_metallib
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attention_minference.metal"
        "omlx_m5_attention_minference.metallib"
        "27.0"
        FAST_MATH
        EXTRA_FLAGS -std=metal4.1)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_chunk_nax.metal")
      omlx_m5_compile_metallib(
        omlx_m5_gdn_chunk_nax_metallib
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_chunk_nax.metal"
        "omlx_m5_gdn_chunk_nax.metallib"
        "27.0"
        EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    endif()
  endif()

  if(OMLX_M5_BUILD_MLX_LEGACY_SMOKE)
    if(NOT Python_EXECUTABLE)
      find_package(Python 3.11 COMPONENTS Interpreter REQUIRED)
    endif()
    execute_process(
      COMMAND
        "${Python_EXECUTABLE}" -c
        "import mlx; print(next(iter(mlx.__path__)))"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE OMLX_M5_MLX_PACKAGE_DIR
      RESULT_VARIABLE OMLX_M5_MLX_QUERY_RESULT)
    if(NOT OMLX_M5_MLX_QUERY_RESULT EQUAL 0)
      message(FATAL_ERROR "Unable to locate the MLX package include directory")
    endif()
    set(OMLX_M5_MLX_INCLUDE_DIR "${OMLX_M5_MLX_PACKAGE_DIR}/include")

    omlx_m5_compile_metallib(
      omlx_m5_nax_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/nax_qmm_smoke.metal"
      "omlx_m5_nax_smoke.metallib"
      "26.2"
      INCLUDE_DIRS "${OMLX_M5_MLX_INCLUDE_DIR}"
      EXTRA_FLAGS
        -Wno-c++17-extensions
        -Wno-c++20-extensions)

    omlx_m5_compile_metallib(
      omlx_m5_mxfp4_tensorops_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mxfp4_tensorops.metal"
      "omlx_m5_mxfp4_tensorops.metallib"
      "26.2"
      INCLUDE_DIRS "${OMLX_M5_MLX_INCLUDE_DIR}"
      EXTRA_FLAGS
        -Wno-c++17-extensions
        -Wno-c++20-extensions)
  endif()
endif()

set(_omlx_m5_row_exact_deployment_target "15.0")
if(OMLX_M5_BUILD_NAX_METALLIB AND
   OMLX_M5_MACOS_SDK_VERSION VERSION_GREATER_EQUAL 27.0)
  set(_omlx_m5_row_exact_deployment_target "27.0")
endif()
omlx_m5_compile_metallib(
  omlx_m5_row_exact_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/row_exact.metal"
  "omlx_m5_row_exact.metallib"
  "${_omlx_m5_row_exact_deployment_target}"
  EXTRA_FLAGS -Wno-c++17-extensions
  EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qmv_t1.metal")

set(OMLX_M5_MXFP4_TENSOROPS_METALLIB_TEST_PATH "")
if(TARGET omlx_m5_mxfp4_tensorops_metallib)
  set(
    OMLX_M5_MXFP4_TENSOROPS_METALLIB_TEST_PATH
    "${omlx_m5_mxfp4_tensorops_metallib_PATH}")
endif()

add_executable(
  omlx_m5_b2_mxfp4_small_t_tests
  tests/b2_mxfp4_small_t_tests.cpp)
target_compile_definitions(
  omlx_m5_b2_mxfp4_small_t_tests
  PRIVATE
  OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH="${omlx_m5_mxfp4_small_t_metallib_PATH}"
  OMLX_M5_MXFP4_TENSOROPS_METALLIB_PATH="${OMLX_M5_MXFP4_TENSOROPS_METALLIB_TEST_PATH}")
target_link_libraries(
  omlx_m5_b2_mxfp4_small_t_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b2_mxfp4_small_t_tests
  omlx_m5_mxfp4_small_t_metallib)
if(TARGET omlx_m5_mxfp4_tensorops_metallib)
  add_dependencies(
    omlx_m5_b2_mxfp4_small_t_tests
    omlx_m5_mxfp4_tensorops_metallib)
endif()
add_test(
  NAME omlx_m5_b2_mxfp4_small_t
  COMMAND omlx_m5_b2_mxfp4_small_t_tests)

add_executable(
  omlx_m5_b2_qmm_broad_tests
  tests/b2_qmm_broad_tests.cpp)
target_compile_definitions(
  omlx_m5_b2_qmm_broad_tests
  PRIVATE
  OMLX_M5_QMM_BROAD_METALLIB_PATH="${omlx_m5_qmm_broad_metallib_PATH}"
  OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH="${omlx_m5_mxfp4_small_t_metallib_PATH}"
  OMLX_M5_MXFP4_TENSOROPS_METALLIB_PATH="${OMLX_M5_MXFP4_TENSOROPS_METALLIB_TEST_PATH}")
target_link_libraries(
  omlx_m5_b2_qmm_broad_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b2_qmm_broad_tests
  omlx_m5_qmm_broad_metallib
  omlx_m5_mxfp4_small_t_metallib)
if(TARGET omlx_m5_mxfp4_tensorops_metallib)
  add_dependencies(
    omlx_m5_b2_qmm_broad_tests
    omlx_m5_mxfp4_tensorops_metallib)
endif()
add_test(
  NAME omlx_m5_b2_qmm_broad
  COMMAND omlx_m5_b2_qmm_broad_tests)

add_executable(
  omlx_m5_b2_moe_tests
  tests/b2_moe_tests.cpp)
target_compile_definitions(
  omlx_m5_b2_moe_tests
  PRIVATE
  OMLX_M5_MOE_METALLIB_PATH="${omlx_m5_moe_metallib_PATH}"
  OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH="${omlx_m5_mxfp4_small_t_metallib_PATH}")
target_link_libraries(
  omlx_m5_b2_moe_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b2_moe_tests
  omlx_m5_moe_metallib
  omlx_m5_mxfp4_small_t_metallib)
add_test(
  NAME omlx_m5_b2_moe
  COMMAND omlx_m5_b2_moe_tests)

add_executable(
  omlx_m5_b3_projection_tests
  tests/b3_projection_tests.cpp)
target_compile_definitions(
  omlx_m5_b3_projection_tests
  PRIVATE
  OMLX_M5_QMV_T1_METALLIB_PATH="${omlx_m5_qmv_t1_metallib_PATH}")
target_link_libraries(
  omlx_m5_b3_projection_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b3_projection_tests
  omlx_m5_qmv_t1_metallib)
add_test(
  NAME omlx_m5_b3_projection
  COMMAND omlx_m5_b3_projection_tests)

add_executable(
  omlx_m5_b3_normalization_tests
  tests/b3_normalization_tests.cpp)
target_compile_definitions(
  omlx_m5_b3_normalization_tests
  PRIVATE
  OMLX_M5_RMSNORM_PROJECTION_METALLIB_PATH="${omlx_m5_rmsnorm_projection_metallib_PATH}"
  OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH="${omlx_m5_mxfp4_small_t_metallib_PATH}"
  OMLX_M5_QMV_T1_METALLIB_PATH="${omlx_m5_qmv_t1_metallib_PATH}")
target_link_libraries(
  omlx_m5_b3_normalization_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b3_normalization_tests
  omlx_m5_rmsnorm_projection_metallib
  omlx_m5_mxfp4_small_t_metallib
  omlx_m5_qmv_t1_metallib)
add_test(
  NAME omlx_m5_b3_normalization
  COMMAND omlx_m5_b3_normalization_tests)

add_executable(
  omlx_m5_b3_attention_tests
  tests/b3_attention_tests.cpp)
target_compile_definitions(
  omlx_m5_b3_attention_tests
  PRIVATE
  OMLX_M5_ATTENTION_POSTPROCESS_METALLIB_PATH="${omlx_m5_attention_postprocess_metallib_PATH}")
target_link_libraries(
  omlx_m5_b3_attention_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b3_attention_tests
  omlx_m5_attention_postprocess_metallib)
add_test(
  NAME omlx_m5_b3_attention
  COMMAND omlx_m5_b3_attention_tests)

add_executable(
  omlx_m5_inspect_attention
  apps/inspect_attention.cpp)
target_include_directories(
  omlx_m5_inspect_attention
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_attention
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_attention
  omlx_m5_attention_postprocess_metallib
  omlx_m5_qmv_t1_metallib)

add_executable(
  omlx_m5_b3_mlp_tests
  tests/b3_mlp_tests.cpp)
target_compile_definitions(
  omlx_m5_b3_mlp_tests
  PRIVATE
  OMLX_M5_MLP_METALLIB_PATH="${omlx_m5_mlp_metallib_PATH}"
  OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH="${omlx_m5_mxfp4_small_t_metallib_PATH}"
  OMLX_M5_QMV_T1_METALLIB_PATH="${omlx_m5_qmv_t1_metallib_PATH}")
target_link_libraries(
  omlx_m5_b3_mlp_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b3_mlp_tests
  omlx_m5_mlp_metallib
  omlx_m5_mxfp4_small_t_metallib
  omlx_m5_qmv_t1_metallib)
add_test(
  NAME omlx_m5_b3_mlp
  COMMAND omlx_m5_b3_mlp_tests)

add_executable(
  omlx_m5_b3_gdn_tests
  tests/b3_gdn_tests.cpp)
target_compile_definitions(
  omlx_m5_b3_gdn_tests
  PRIVATE
  OMLX_M5_GDN_LAYER_METALLIB_PATH="${omlx_m5_gdn_layer_metallib_PATH}")
target_link_libraries(
  omlx_m5_b3_gdn_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b3_gdn_tests
  omlx_m5_gdn_layer_metallib)
add_test(
  NAME omlx_m5_b3_gdn
  COMMAND omlx_m5_b3_gdn_tests)

add_executable(
  omlx_m5_b3_vocab_tests
  tests/b3_vocab_tests.cpp)
target_compile_definitions(
  omlx_m5_b3_vocab_tests
  PRIVATE
  OMLX_M5_VOCAB_METALLIB_PATH="${omlx_m5_vocab_metallib_PATH}"
  OMLX_M5_QMV_T1_METALLIB_PATH="${omlx_m5_qmv_t1_metallib_PATH}")
target_link_libraries(
  omlx_m5_b3_vocab_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b3_vocab_tests
  omlx_m5_vocab_metallib
  omlx_m5_qmv_t1_metallib)
add_test(
  NAME omlx_m5_b3_vocab
  COMMAND omlx_m5_b3_vocab_tests)

add_executable(
  omlx_m5_b3_schedule_tests
  tests/b3_schedule_tests.cpp)
target_compile_definitions(
  omlx_m5_b3_schedule_tests
  PRIVATE
  OMLX_M5_CLASSIC_METALLIB_PATH="${omlx_m5_classic_metallib_PATH}")
target_link_libraries(
  omlx_m5_b3_schedule_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b3_schedule_tests
  omlx_m5_classic_metallib)
add_test(
  NAME omlx_m5_b3_schedule
  COMMAND omlx_m5_b3_schedule_tests)

add_executable(
  omlx_m5_b4_icb_canary_tests
  tests/b4_icb_canary_tests.cpp)
target_compile_definitions(
  omlx_m5_b4_icb_canary_tests
  PRIVATE
  OMLX_M5_ICB_CANARY_METALLIB_PATH="${omlx_m5_icb_canary_metallib_PATH}")
target_link_libraries(
  omlx_m5_b4_icb_canary_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b4_icb_canary_tests
  omlx_m5_icb_canary_metallib)
add_test(
  NAME omlx_m5_b4_icb_canary
  COMMAND omlx_m5_b4_icb_canary_tests)

add_executable(
  omlx_m5_inspect_icb_canary
  apps/inspect_icb_canary.cpp)
target_include_directories(
  omlx_m5_inspect_icb_canary
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_definitions(
  omlx_m5_inspect_icb_canary
  PRIVATE
  OMLX_M5_ICB_CANARY_METALLIB_PATH="${omlx_m5_icb_canary_metallib_PATH}")
target_link_libraries(
  omlx_m5_inspect_icb_canary
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_icb_canary
  omlx_m5_icb_canary_metallib)

add_executable(
  omlx_m5_b4_parameter_abi_tests
  tests/b4_parameter_abi_tests.cpp)
target_compile_definitions(
  omlx_m5_b4_parameter_abi_tests
  PRIVATE
  OMLX_M5_ICB_CANARY_METALLIB_PATH="${omlx_m5_icb_canary_metallib_PATH}")
target_link_libraries(
  omlx_m5_b4_parameter_abi_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b4_parameter_abi_tests
  omlx_m5_icb_canary_metallib)
add_test(
  NAME omlx_m5_b4_parameter_abi
  COMMAND omlx_m5_b4_parameter_abi_tests)

add_executable(
  omlx_m5_inspect_parameter_abi
  apps/inspect_parameter_abi.cpp)
target_include_directories(
  omlx_m5_inspect_parameter_abi
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_definitions(
  omlx_m5_inspect_parameter_abi
  PRIVATE
  OMLX_M5_ICB_CANARY_METALLIB_PATH="${omlx_m5_icb_canary_metallib_PATH}")
target_link_libraries(
  omlx_m5_inspect_parameter_abi
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_parameter_abi
  omlx_m5_icb_canary_metallib)

add_executable(
  omlx_m5_b4_topology_profile_tests
  tests/b4_topology_profile_tests.cpp)
target_link_libraries(
  omlx_m5_b4_topology_profile_tests
  PRIVATE
  omlx_m5_runtime)
add_test(
  NAME omlx_m5_b4_topology_profile
  COMMAND omlx_m5_b4_topology_profile_tests)

add_executable(
  omlx_m5_inspect_topology_profiles
  apps/inspect_topology_profiles.cpp)
target_include_directories(
  omlx_m5_inspect_topology_profiles
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_topology_profiles
  PRIVATE
  omlx_m5_runtime)

if(NOT Python_EXECUTABLE)
  find_package(Python 3.11 COMPONENTS Interpreter REQUIRED)
endif()
set(OMLX_M5_HAS_NUMPY OFF)
if(OMLX_M5_BUILD_MLX_FIXTURES)
  set(OMLX_M5_HAS_NUMPY ON)
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy"
    RESULT_VARIABLE _omlx_m5_numpy_result
    OUTPUT_QUIET
    ERROR_QUIET)
  if(_omlx_m5_numpy_result EQUAL 0)
    set(OMLX_M5_HAS_NUMPY ON)
  else()
    message(
      STATUS
      "numpy not found in ${Python_EXECUTABLE}; skipping B4 Python fixture tests")
  endif()
endif()
set(OMLX_M5_B4_PROGRAM_FIXTURE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/b4-program")
set(OMLX_M5_B4_PROGRAM_FIXTURE
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}/fixture.omlxm5")
set(OMLX_M5_B4_PROGRAM_FIXTURE_CONFIG
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}/source/config.json")
set(OMLX_M5_B4_PROGRAM_FIXTURE_STAMP
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}/fixture.stamp")
if(OMLX_M5_HAS_NUMPY)
  add_custom_command(
    OUTPUT
    "${OMLX_M5_B4_PROGRAM_FIXTURE_STAMP}"
    BYPRODUCTS
    "${OMLX_M5_B4_PROGRAM_FIXTURE}"
    "${OMLX_M5_B4_PROGRAM_FIXTURE_CONFIG}"
    COMMAND
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_b4_program_fixture.py"
    --output-dir
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}"
    DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_b4_program_fixture.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/m5_artifact.py"
    COMMENT "Generating the B4 Qwen Program fixture"
    VERBATIM)
  add_custom_target(
    omlx_m5_b4_program_fixture
    DEPENDS "${OMLX_M5_B4_PROGRAM_FIXTURE_STAMP}")

  add_executable(
    omlx_m5_b4_qwen_program_tests
    tests/b4_qwen_program_tests.cpp)
  target_link_libraries(
    omlx_m5_b4_qwen_program_tests
    PRIVATE
    omlx_m5_runtime)
  target_compile_definitions(
    omlx_m5_b4_qwen_program_tests
    PRIVATE
    OMLX_M5_QWEN_AR_METALLIB_PATH="${omlx_m5_qwen_ar_metallib_PATH}")
  add_dependencies(
    omlx_m5_b4_qwen_program_tests
    omlx_m5_b4_program_fixture
    omlx_m5_qwen_ar_metallib)
  add_test(
    NAME omlx_m5_b4_qwen_program
    COMMAND
    omlx_m5_b4_qwen_program_tests
    "${OMLX_M5_B4_PROGRAM_FIXTURE}"
    "${OMLX_M5_B4_PROGRAM_FIXTURE_CONFIG}")

  add_executable(
    omlx_m5_b4_ar_tests
    tests/b4_ar_tests.cpp)
  target_compile_definitions(
    omlx_m5_b4_ar_tests
    PRIVATE
    OMLX_M5_QWEN_AR_METALLIB_PATH="${omlx_m5_qwen_ar_metallib_PATH}")
  target_link_libraries(
    omlx_m5_b4_ar_tests
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_b4_ar_tests
    omlx_m5_b4_program_fixture
    omlx_m5_qwen_ar_metallib)
  add_test(
    NAME omlx_m5_b4_ar
    COMMAND
    omlx_m5_b4_ar_tests
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}")
endif()

set(OMLX_M5_B5_DFLASH2_FIXTURE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/b5-dflash2")
set(OMLX_M5_B5_DFLASH2_FIXTURE
    "${OMLX_M5_B5_DFLASH2_FIXTURE_DIR}/dflash2-fixture.omlxm5")
set(OMLX_M5_B5_DFLASH2_FIXTURE_CONFIG
    "${OMLX_M5_B5_DFLASH2_FIXTURE_DIR}/source/config.json")
set(OMLX_M5_B5_DFLASH2_ORACLE
    "${OMLX_M5_B5_DFLASH2_FIXTURE_DIR}/oracle.json")
set(OMLX_M5_B5_DFLASH2_STAMP_PATH
    "${OMLX_M5_B5_DFLASH2_FIXTURE_DIR}/fixture.stamp")
set(OMLX_M5_B5_DFLASH2_FIXTURE_STAMP "${OMLX_M5_B5_DFLASH2_STAMP_PATH}")

set(OMLX_M5_B5_REAL_DFLASH2_DIR
    "${CMAKE_SOURCE_DIR}/models/z-lab/Qwen3.8-27B-DFlash2"
    CACHE PATH
    "Real Qwen3.8-27B DFlash2 companion checkpoint used by B5 integration tests")
set(OMLX_M5_B5_REAL_DFLASH2_ARTIFACT_DIR
    "${CMAKE_SOURCE_DIR}/.cache/b5/real")
set(OMLX_M5_B5_REAL_DFLASH2_ARTIFACT
    "${OMLX_M5_B5_REAL_DFLASH2_ARTIFACT_DIR}/qwen38-27b-dflash2.omlxm5")

if(OMLX_M5_BUILD_MLX_FIXTURES)
  add_executable(
    omlx_m5_b5_dflash_program_tests
    tests/b5_dflash_program_tests.cpp)
  target_compile_definitions(
    omlx_m5_b5_dflash_program_tests
    PRIVATE
    OMLX_M5_QWEN_AR_METALLIB_PATH="${omlx_m5_qwen_ar_metallib_PATH}"
    OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH="${omlx_m5_dflash2_program_metallib_PATH}")
  target_link_libraries(
    omlx_m5_b5_dflash_program_tests
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_b5_dflash_program_tests
    omlx_m5_b4_program_fixture
    omlx_m5_qwen_ar_metallib)
  add_test(
    NAME omlx_m5_b5_dflash_program
    COMMAND
    omlx_m5_b5_dflash_program_tests
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}"
    "${OMLX_M5_B5_DFLASH2_FIXTURE}"
    "${OMLX_M5_B5_DFLASH2_FIXTURE_CONFIG}")

  add_custom_command(
    OUTPUT "${OMLX_M5_B5_DFLASH2_FIXTURE_STAMP}"
    BYPRODUCTS
    "${OMLX_M5_B5_DFLASH2_FIXTURE}"
    "${OMLX_M5_B5_DFLASH2_FIXTURE_CONFIG}"
    "${OMLX_M5_B5_DFLASH2_ORACLE}"
    COMMAND
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_b5_dflash2_fixture.py"
    --output-dir
    "${OMLX_M5_B5_DFLASH2_FIXTURE_DIR}"
    DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_b5_dflash2_fixture.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/m5_artifact.py"
    COMMENT "Generating the B5 true DFlash2 fixture and dflash-mlx oracle"
    VERBATIM)
  add_custom_target(
    omlx_m5_b5_dflash2_fixture
    DEPENDS "${OMLX_M5_B5_DFLASH2_FIXTURE_STAMP}")

  add_dependencies(
    omlx_m5_b5_dflash_program_tests
    omlx_m5_b5_dflash2_fixture
    omlx_m5_dflash2_program_metallib)

  add_executable(
    omlx_m5_b5_dflash2_program_tests
    tests/b5_dflash2_program_tests.cpp)
  target_include_directories(
    omlx_m5_b5_dflash2_program_tests
    PRIVATE
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_definitions(
    omlx_m5_b5_dflash2_program_tests
    PRIVATE
    OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH="${omlx_m5_dflash2_program_metallib_PATH}")
  target_link_libraries(
    omlx_m5_b5_dflash2_program_tests
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_b5_dflash2_program_tests
    omlx_m5_b5_dflash2_fixture
    omlx_m5_dflash2_program_metallib)

  if(EXISTS "${OMLX_M5_B5_REAL_DFLASH2_DIR}/config.json" AND
     EXISTS "${OMLX_M5_B5_REAL_DFLASH2_DIR}/model.safetensors" AND
     EXISTS "${OMLX_M5_B5_REAL_DFLASH2_DIR}/REVISION")
    add_custom_command(
      OUTPUT "${OMLX_M5_B5_REAL_DFLASH2_ARTIFACT}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${OMLX_M5_B5_REAL_DFLASH2_ARTIFACT_DIR}"
      COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/m5_artifact.py"
      convert
      --source "${OMLX_M5_B5_REAL_DFLASH2_DIR}"
      --output "${OMLX_M5_B5_REAL_DFLASH2_ARTIFACT}"
      --source-repo z-lab/Qwen3.8-27B-DFlash2
      --expected-revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
      --profile dflash2_bf16
      DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/m5_artifact.py"
      "${OMLX_M5_B5_REAL_DFLASH2_DIR}/config.json"
      "${OMLX_M5_B5_REAL_DFLASH2_DIR}/model.safetensors"
      "${OMLX_M5_B5_REAL_DFLASH2_DIR}/REVISION"
      COMMENT "Converting the real Qwen3.8-27B DFlash2 companion artifact"
      VERBATIM)
    add_custom_target(
      omlx_m5_b5_real_dflash2_artifact
      DEPENDS "${OMLX_M5_B5_REAL_DFLASH2_ARTIFACT}")
    add_dependencies(
      omlx_m5_b5_dflash2_program_tests
      omlx_m5_b5_real_dflash2_artifact)
    add_test(
      NAME omlx_m5_b5_dflash2_program
      COMMAND
      omlx_m5_b5_dflash2_program_tests
      "${OMLX_M5_B5_DFLASH2_FIXTURE}"
      "${OMLX_M5_B5_DFLASH2_FIXTURE_CONFIG}"
      "${OMLX_M5_B5_DFLASH2_ORACLE}"
      "${OMLX_M5_B5_REAL_DFLASH2_ARTIFACT}"
      "${OMLX_M5_B5_REAL_DFLASH2_DIR}/config.json")
  else()
    message(
      STATUS
      "B5 real DFlash2 companion checkpoint not found at ${OMLX_M5_B5_REAL_DFLASH2_DIR}; real-companion test is unavailable")
  endif()

  add_executable(
    omlx_m5_inspect_dflash_program
    apps/inspect_dflash_program.cpp)
  target_include_directories(
    omlx_m5_inspect_dflash_program
    PRIVATE
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_definitions(
    omlx_m5_inspect_dflash_program
    PRIVATE
    OMLX_M5_QWEN_AR_METALLIB_PATH="${omlx_m5_qwen_ar_metallib_PATH}"
    OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH="${omlx_m5_dflash2_program_metallib_PATH}")
  target_link_libraries(
    omlx_m5_inspect_dflash_program
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_inspect_dflash_program
    omlx_m5_b4_program_fixture
    omlx_m5_b5_dflash2_fixture
    omlx_m5_qwen_ar_metallib
    omlx_m5_dflash2_program_metallib)
endif()

add_executable(
  m5rt_runner
  apps/m5rt_runner.cpp)
target_include_directories(
  m5rt_runner
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_definitions(
  m5rt_runner
  PRIVATE
  OMLX_M5_QWEN_AR_METALLIB_PATH="${omlx_m5_qwen_ar_metallib_PATH}")
target_link_libraries(
  m5rt_runner
  PRIVATE
  omlx_m5_core_runtime)
add_dependencies(
  m5rt_runner
  omlx_m5_qwen_ar_metallib)

if(OMLX_M5_BUILD_MLX_FIXTURES)
  add_executable(
    omlx_m5_b6_native_backend_tests
    tests/b6_native_backend_tests.cpp)
  target_link_libraries(
    omlx_m5_b6_native_backend_tests
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_b6_native_backend_tests
    omlx_m5_b4_program_fixture
    omlx_m5_b5_dflash2_fixture
    omlx_m5_qwen_ar_metallib
    omlx_m5_dflash2_program_metallib)
  add_test(
    NAME omlx_m5_b6_native_backend
    COMMAND
    omlx_m5_b6_native_backend_tests
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}"
    "${omlx_m5_qwen_ar_metallib_PATH}"
    "${OMLX_M5_B5_DFLASH2_FIXTURE}"
    "${OMLX_M5_B5_DFLASH2_FIXTURE_CONFIG}"
    "${omlx_m5_dflash2_program_metallib_PATH}")
endif()

add_executable(
  mseries_backendd
  apps/mseries_backendd.cpp)
target_include_directories(
  mseries_backendd
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_definitions(
  mseries_backendd
  PRIVATE
  NINFER_METAL_DISPLAY_VERSION="${NINFER_METAL_DISPLAY_VERSION}"
  NINFER_METAL_CODENAME="${NINFER_METAL_CODENAME}"
  NINFER_METAL_SEMVER="${NINFER_METAL_SEMVER}"
  NINFER_METAL_CORE_ABI=${NINFER_METAL_CORE_ABI})
target_link_libraries(
  mseries_backendd
  PRIVATE
  omlx_m5_core_runtime)
add_dependencies(
  mseries_backendd
  omlx_m5_qwen_ar_metallib
  omlx_m5_dflash2_program_metallib)

if(OMLX_M5_BUILD_MLX_FIXTURES)
  add_test(
    NAME omlx_m5_provider_sidecar
    COMMAND
    ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/provider_sidecar_smoke.py
    $<TARGET_FILE:mseries_backendd>
    "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}"
    "${omlx_m5_qwen_ar_metallib_PATH}"
    "${OMLX_M5_B5_DFLASH2_FIXTURE}"
    "${OMLX_M5_B5_DFLASH2_FIXTURE_CONFIG}"
    "${omlx_m5_dflash2_program_metallib_PATH}")
endif()

if(OMLX_M5_HAS_NUMPY)
  add_executable(
    omlx_m5_inspect_qwen_program
    apps/inspect_qwen_program.cpp)
  target_include_directories(
    omlx_m5_inspect_qwen_program
    PRIVATE
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(
    omlx_m5_inspect_qwen_program
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_inspect_qwen_program
    omlx_m5_b4_program_fixture)
endif()

add_executable(
  omlx_m5_inspect_layer_schedule
  apps/inspect_layer_schedule.cpp)
target_include_directories(
  omlx_m5_inspect_layer_schedule
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_layer_schedule
  PRIVATE
  omlx_m5_runtime)

add_executable(
  omlx_m5_inspect_command_batch
  apps/inspect_command_batch.cpp)
target_include_directories(
  omlx_m5_inspect_command_batch
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_definitions(
  omlx_m5_inspect_command_batch
  PRIVATE
  OMLX_M5_CLASSIC_METALLIB_PATH="${omlx_m5_classic_metallib_PATH}")
target_link_libraries(
  omlx_m5_inspect_command_batch
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_command_batch
  omlx_m5_classic_metallib)

add_executable(
  omlx_m5_inspect_vocab
  apps/inspect_vocab.cpp)
target_include_directories(
  omlx_m5_inspect_vocab
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_vocab
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_vocab
  omlx_m5_vocab_metallib
  omlx_m5_qmv_t1_metallib
  omlx_m5_rmsnorm_projection_metallib)

add_executable(
  omlx_m5_inspect_gdn
  apps/inspect_gdn.cpp)
target_include_directories(
  omlx_m5_inspect_gdn
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_gdn
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_gdn
  omlx_m5_gdn_layer_metallib
  omlx_m5_qmv_t1_metallib)

add_executable(
  omlx_m5_inspect_mlp
  apps/inspect_mlp.cpp)
target_include_directories(
  omlx_m5_inspect_mlp
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_mlp
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_mlp
  omlx_m5_mlp_metallib
  omlx_m5_mxfp4_small_t_metallib
  omlx_m5_qmv_t1_metallib
  omlx_m5_moe_metallib)

add_executable(
  omlx_m5_inspect_normalization
  apps/inspect_normalization.cpp)
target_include_directories(
  omlx_m5_inspect_normalization
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_normalization
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_normalization
  omlx_m5_rmsnorm_projection_metallib
  omlx_m5_mxfp4_small_t_metallib
  omlx_m5_qmv_t1_metallib)

add_executable(
  omlx_m5_inspect_projection
  apps/inspect_projection.cpp)
target_include_directories(
  omlx_m5_inspect_projection
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_projection
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_projection
  omlx_m5_qmv_t1_metallib)

add_executable(
  omlx_m5_inspect_moe
  apps/inspect_moe.cpp)
target_include_directories(
  omlx_m5_inspect_moe
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_moe
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_moe
  omlx_m5_moe_metallib
  omlx_m5_mxfp4_small_t_metallib)

if(TARGET omlx_m5_mxfp4_tensorops_metallib)
  add_executable(
    omlx_m5_inspect_mxfp4_small_t
    apps/inspect_mxfp4_small_t.cpp)
  target_include_directories(
    omlx_m5_inspect_mxfp4_small_t
    PRIVATE
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(
    omlx_m5_inspect_mxfp4_small_t
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_inspect_mxfp4_small_t
    omlx_m5_mxfp4_small_t_metallib
    omlx_m5_mxfp4_tensorops_metallib)

  add_executable(
    omlx_m5_inspect_qmm_broad
    apps/inspect_qmm_broad.cpp)
  target_compile_definitions(
    omlx_m5_inspect_qmm_broad
    PRIVATE
    OMLX_M5_QMM_BROAD_METALLIB_PATH="${omlx_m5_qmm_broad_metallib_PATH}"
    OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH="${omlx_m5_mxfp4_small_t_metallib_PATH}"
    OMLX_M5_MXFP4_TENSOROPS_METALLIB_PATH="${omlx_m5_mxfp4_tensorops_metallib_PATH}")
  target_include_directories(
    omlx_m5_inspect_qmm_broad
    PRIVATE
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(
    omlx_m5_inspect_qmm_broad
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(
    omlx_m5_inspect_qmm_broad
    omlx_m5_qmm_broad_metallib
    omlx_m5_mxfp4_small_t_metallib
    omlx_m5_mxfp4_tensorops_metallib)
endif()

add_executable(
  omlx_m5_b1_registry_tests
  tests/b1_registry_tests.cpp)
target_compile_definitions(
  omlx_m5_b1_registry_tests
  PRIVATE
  OMLX_M5_CLASSIC_METALLIB_PATH="${omlx_m5_classic_metallib_PATH}")
target_link_libraries(
  omlx_m5_b1_registry_tests
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_b1_registry_tests
  omlx_m5_classic_metallib)
add_test(
  NAME omlx_m5_b1_registry
  COMMAND omlx_m5_b1_registry_tests)

add_executable(
  omlx_m5_inspect_runtime
  apps/inspect_runtime.cpp)
target_include_directories(
  omlx_m5_inspect_runtime
  PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_inspect_runtime
  PRIVATE
  omlx_m5_runtime)
add_dependencies(
  omlx_m5_inspect_runtime
  omlx_m5_classic_metallib)

if(OMLX_M5_BUILD_PROBES)
  set(_omlx_m5_probe_nax_path "${omlx_m5_nax_metallib_PATH}")
  if(NOT _omlx_m5_probe_nax_path AND TARGET omlx_m5_mxfp4_nax_metallib)
    set(_omlx_m5_probe_nax_path "${omlx_m5_mxfp4_nax_metallib_PATH}")
  endif()
  add_executable(
    omlx_m5_probe
    probes/m5_probe.mm)
  target_compile_options(
    omlx_m5_probe
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  target_compile_definitions(
    omlx_m5_probe
    PRIVATE
    OMLX_M5_CLASSIC_METALLIB_PATH="${omlx_m5_classic_metallib_PATH}"
    OMLX_M5_NAX_METALLIB_PATH="${_omlx_m5_probe_nax_path}")
  target_link_libraries(
    omlx_m5_probe
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(omlx_m5_probe omlx_m5_classic_metallib)
  if(TARGET omlx_m5_nax_metallib)
    add_dependencies(omlx_m5_probe omlx_m5_nax_metallib)
  elseif(TARGET omlx_m5_mxfp4_nax_metallib)
    add_dependencies(omlx_m5_probe omlx_m5_mxfp4_nax_metallib)
  endif()
  add_test(
    NAME omlx_m5_probe_smoke
    COMMAND omlx_m5_probe --bytes 4194304 --iterations 2)
  add_executable(
    omlx_m5_d3_gdn_probe
    probes/d3_gdn_probe.mm)
  target_compile_options(
    omlx_m5_d3_gdn_probe
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  target_compile_definitions(
    omlx_m5_d3_gdn_probe
    PRIVATE
    OMLX_M5_D3_METALLIB_PATH="${omlx_m5_d3_metallib_PATH}")
  target_link_libraries(
    omlx_m5_d3_gdn_probe
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(omlx_m5_d3_gdn_probe omlx_m5_d3_metallib)
  add_test(
    NAME omlx_m5_d3_gdn_metal
    COMMAND omlx_m5_d3_gdn_probe)
  add_executable(
    omlx_m5_d2_reference_probe
    probes/d2_reference_probe.mm)
  target_compile_options(
    omlx_m5_d2_reference_probe
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  target_compile_definitions(
    omlx_m5_d2_reference_probe
    PRIVATE
    OMLX_M5_D2_METALLIB_PATH="${omlx_m5_d2_metallib_PATH}")
  target_link_libraries(
    omlx_m5_d2_reference_probe
    PRIVATE
    omlx_m5_runtime)
  add_dependencies(omlx_m5_d2_reference_probe omlx_m5_d2_metallib)
  add_test(
    NAME omlx_m5_d2_reference_metal
    COMMAND omlx_m5_d2_reference_probe)
  add_executable(
    omlx_m5_nax_qmm_probe
    probes/nax_qmm_probe.mm)
  target_compile_options(
    omlx_m5_nax_qmm_probe
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  target_compile_definitions(
    omlx_m5_nax_qmm_probe
    PRIVATE
    OMLX_M5_NAX_METALLIB_PATH="${omlx_m5_nax_metallib_PATH}")
  target_link_libraries(
    omlx_m5_nax_qmm_probe
    PRIVATE
    omlx_m5_runtime)
  if(TARGET omlx_m5_nax_metallib)
    add_dependencies(omlx_m5_nax_qmm_probe omlx_m5_nax_metallib)
  endif()
  add_test(
    NAME omlx_m5_nax_qmm_probe_boundary
    COMMAND
      omlx_m5_nax_qmm_probe
      --m 1
      --n 64
      --k 64
      --bits 5
      --iterations 1)
  add_executable(
    omlx_m5_hw_counter_probe
    tools/hw_counter_probe.cpp)
  target_include_directories(
    omlx_m5_hw_counter_probe
    PRIVATE
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(
    omlx_m5_hw_counter_probe
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  target_compile_definitions(
    omlx_m5_hw_counter_probe
    PRIVATE
    OMLX_M5_CLASSIC_METALLIB_PATH="${omlx_m5_classic_metallib_PATH}"
    OMLX_M5_NAX_METALLIB_PATH="${omlx_m5_nax_metallib_PATH}")
  target_link_libraries(
    omlx_m5_hw_counter_probe
    PRIVATE
    omlx_m5_core_runtime)
  add_dependencies(omlx_m5_hw_counter_probe omlx_m5_classic_metallib)
  if(TARGET omlx_m5_nax_metallib)
    add_dependencies(omlx_m5_hw_counter_probe omlx_m5_nax_metallib)
  endif()
  add_test(
    NAME omlx_m5_hw_counter_probe_smoke
    COMMAND omlx_m5_hw_counter_probe --mode enumerate --iterations 2)
endif()

if(OMLX_M5_BUILD_PYTHON)
  find_package(
    Python 3.11
    COMPONENTS Interpreter Development.Module
    REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE OMLX_M5_NANOBIND_RESULT)
  if(NOT OMLX_M5_NANOBIND_RESULT EQUAL 0)
    message(
      FATAL_ERROR
      "nanobind is required for OMLX_M5_BUILD_PYTHON=ON; install nanobind==2.15.0 in ${Python_EXECUTABLE}")
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(
    omlx_m5_native
    NB_STATIC
    STABLE_ABI
    bindings/python_module.cpp)
  target_link_libraries(
    omlx_m5_native
    PRIVATE
    omlx_m5_core_runtime)
  add_test(
    NAME omlx_m5_python_binding
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_binding_smoke.py"
      "$<TARGET_FILE_DIR:omlx_m5_native>")
  add_dependencies(
    omlx_m5_native
    omlx_m5_qwen_ar_metallib
    omlx_m5_dflash2_program_metallib)
  if(OMLX_M5_BUILD_MLX_FIXTURES)
    add_dependencies(
      omlx_m5_native
      omlx_m5_b4_program_fixture
      omlx_m5_b5_dflash2_fixture)
    add_test(
      NAME omlx_m5_python_native_binding
      COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_binding_native.py"
        "$<TARGET_FILE_DIR:omlx_m5_native>"
        "${OMLX_M5_B4_PROGRAM_FIXTURE_DIR}"
        "${omlx_m5_qwen_ar_metallib_PATH}"
        "${OMLX_M5_B5_DFLASH2_FIXTURE}"
        "${OMLX_M5_B5_DFLASH2_FIXTURE_CONFIG}"
        "${omlx_m5_dflash2_program_metallib_PATH}")
  endif()
  add_test(
    NAME omlx_m5_benchmark_schema
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/m5_bench_validate.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/example.result.json")
  add_test(
    NAME omlx_m5_d0_assets
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_d0_assets.py")
  add_test(
    NAME omlx_m5_b0_artifact_roundtrip
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/b0_artifact_roundtrip.py"
      --cpp-inspector
      "$<TARGET_FILE:omlx_m5_inspect_artifact>"
      --golden
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/b0_golden_vectors.json")
  set_tests_properties(
    omlx_m5_b0_artifact_roundtrip
    PROPERTIES
    TIMEOUT 120)
  add_test(
    NAME omlx_m5_b1_materialization_roundtrip
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/b1_materialization_roundtrip.py"
      --inspector
      "$<TARGET_FILE:omlx_m5_inspect_materialization>")
  set_tests_properties(
    omlx_m5_b1_materialization_roundtrip
    PROPERTIES
    TIMEOUT 120)
  add_test(
    NAME omlx_m5_qmm_sweep_plan
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/m5_qmm_sweep.py"
      --manifest
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/qwen38_27b_oq4e_shapes.json")
  if(TARGET omlx_m5_d1_model_integration_tests)
    add_test(
      NAME omlx_m5_d1_quantization_parity
      COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/d1_quantization_parity.py"
        "$<TARGET_FILE:omlx_m5_inspect_model>"
        "${OMLX_M5_MODEL_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/Jundot__Qwen3.8-27B-oQ4e-mtp.headers.json")
    add_test(
      NAME omlx_m5_d2_real_projection_parity
      COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/d2_real_projection_parity.py"
        "$<TARGET_FILE:omlx_m5_reference_projection>"
        "${OMLX_M5_MODEL_DIR}")
    set_tests_properties(
      omlx_m5_d2_real_projection_parity
      PROPERTIES
      LABELS "integration")
    add_test(
      NAME omlx_m5_d2_real_layer_parity
      COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/d2_real_layer_parity.py"
        "$<TARGET_FILE:omlx_m5_reference_layer>"
        "${OMLX_M5_MODEL_DIR}")
    set_tests_properties(
      omlx_m5_d2_real_layer_parity
      PROPERTIES
      LABELS "integration"
      TIMEOUT 300)
  endif()
  if(EXISTS "${OMLX_M5_SMALL_MODEL_0_8B_DIR}/model.safetensors" AND
     EXISTS "${OMLX_M5_SMALL_MODEL_4B_DIR}/model.safetensors")
    add_executable(
      omlx_m5_d2_model_binding_tests
      tests/d2_model_binding_tests.cpp)
    target_link_libraries(
      omlx_m5_d2_model_binding_tests
      PRIVATE
      omlx_m5_runtime)
    add_test(
      NAME omlx_m5_d2_model_bindings
      COMMAND
        omlx_m5_d2_model_binding_tests
        "${OMLX_M5_SMALL_MODEL_0_8B_DIR}"
        "${OMLX_M5_SMALL_MODEL_4B_DIR}")
    add_test(
      NAME omlx_m5_d2_small_model_e2e
      COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/d2_small_model_e2e.py"
        "${OMLX_M5_SMALL_MODEL_0_8B_DIR}"
        "${OMLX_M5_SMALL_MODEL_4B_DIR}")
    set_tests_properties(
      omlx_m5_d2_small_model_e2e
      PROPERTIES
      LABELS "integration"
      ENVIRONMENT "OMLX_WITH_CUSTOM_KERNEL=0"
      TIMEOUT 600)
  else()
    message(
      STATUS
      "D2 Qwen3.5 0.8B/4B fixtures are incomplete; small-model end-to-end test is unavailable")
  endif()
endif()

# Full-model mathematical qualification, separate from the historical control fixture.
add_custom_target(omlx_m5_qwen_forward_resources)
add_dependencies(omlx_m5_qwen_forward_resources
  omlx_m5_attention_exact_metallib
  omlx_m5_position_sampling_metallib
  omlx_m5_row_exact_metallib
  omlx_m5_dflash_forward_metallib
  omlx_m5_attention_split_kv_metallib
  omlx_m5_target_verify_gpu_metallib
  omlx_m5_quantized_matrix_metallib
  omlx_m5_qwen_forward_metallib omlx_m5_vocab_metallib omlx_m5_qmv_t1_metallib
  omlx_m5_qmm_small_t_metallib omlx_m5_qmm_broad_metallib omlx_m5_mxfp4_small_t_metallib
  omlx_m5_rmsnorm_projection_metallib omlx_m5_attention_postprocess_metallib
  omlx_m5_gdn_layer_metallib omlx_m5_mlp_metallib omlx_m5_moe_metallib)
if(TARGET omlx_m5_mxfp4_nax_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_mxfp4_nax_metallib)
endif()
if(TARGET omlx_m5_affine_nax_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_affine_nax_metallib)
endif()
if(TARGET omlx_m5_bf16_nax_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_bf16_nax_metallib)
endif()
if(TARGET omlx_m5_moe_nax_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_moe_nax_metallib)
endif()
if(TARGET omlx_m5_attention_affine4_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_attention_affine4_metallib)
endif()
if(TARGET omlx_m5_attention_nax_prefill_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_attention_nax_prefill_metallib)
endif()
if(TARGET omlx_m5_attention_sage2_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_attention_sage2_metallib)
endif()
if(TARGET omlx_m5_attention_minference_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_attention_minference_metallib)
endif()
if(TARGET omlx_m5_gdn_chunk_nax_metallib)
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_gdn_chunk_nax_metallib)
endif()

omlx_m5_compile_metallib(omlx_m5_bounded_batch_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/bounded_batch.metal"
  "omlx_m5_bounded_batch.metallib" "15.0")
add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_bounded_batch_metallib)

omlx_m5_compile_metallib(omlx_m5_mtp_forward_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mtp_forward.metal"
  "omlx_m5_mtp_forward.metallib" "15.0")
add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_mtp_forward_metallib)

omlx_m5_compile_metallib(omlx_m5_lora_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/lora.metal"
  "omlx_m5_lora.metallib" "15.0")
add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_lora_metallib)

add_dependencies(mseries_backendd omlx_m5_qwen_forward_resources)
if(TARGET omlx_m5_native)
  add_dependencies(omlx_m5_native omlx_m5_qwen_forward_resources)
endif()
add_dependencies(m5rt_runner omlx_m5_qwen_forward_resources)
if(TARGET omlx_m5_hw_counter_probe)
  add_dependencies(omlx_m5_hw_counter_probe omlx_m5_qwen_forward_resources)
endif()

add_executable(omlx_m5_tf_row_exact_tests tests/tf_row_exact_tests.cpp)
target_include_directories(omlx_m5_tf_row_exact_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_tf_row_exact_tests PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_tf_row_exact_tests omlx_m5_row_exact_metallib
  omlx_m5_quantized_matrix_metallib
  omlx_m5_qmv_t1_metallib omlx_m5_qmm_small_t_metallib omlx_m5_qmm_broad_metallib
  omlx_m5_mxfp4_small_t_metallib omlx_m5_qwen_forward_metallib)
add_test(NAME omlx_m5_p02_tf_row_exact COMMAND $<TARGET_FILE:omlx_m5_tf_row_exact_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")
add_test(NAME omlx_m5_full_model_measurement COMMAND "${Python_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/full_model_measurement_tests.py")

if(OMLX_M5_HAS_NUMPY)
  add_executable(omlx_m5_control_fixture_runner tests/control_fixture_runner.cpp)
  target_include_directories(omlx_m5_control_fixture_runner PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_control_fixture_runner PRIVATE omlx_m5_runtime)
  target_compile_definitions(omlx_m5_control_fixture_runner PRIVATE
    OMLX_M5_QWEN_AR_METALLIB_PATH="${omlx_m5_qwen_ar_metallib_PATH}")
  add_dependencies(omlx_m5_control_fixture_runner omlx_m5_b4_program_fixture omlx_m5_qwen_ar_metallib)
endif()

add_executable(omlx_m5_p03_selector_tests tests/p03_selector_tests.cpp)
target_include_directories(omlx_m5_p03_selector_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_p03_selector_tests PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_p03_selector_tests omlx_m5_dflash_forward_metallib)
add_test(NAME omlx_m5_p03_selector COMMAND $<TARGET_FILE:omlx_m5_p03_selector_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")

add_executable(omlx_m5_tf_position_sampling_tests tests/tf_position_sampling_tests.cpp)
target_include_directories(omlx_m5_tf_position_sampling_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_tf_position_sampling_tests PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_tf_position_sampling_tests omlx_m5_position_sampling_metallib)
add_test(NAME omlx_m5_p03_tf_position COMMAND $<TARGET_FILE:omlx_m5_tf_position_sampling_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")

add_executable(omlx_m5_tf_target_equality_tests tests/tf_target_equality_tests.cpp)
target_include_directories(omlx_m5_tf_target_equality_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_tf_target_equality_tests PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_tf_target_equality_tests omlx_m5_position_sampling_metallib)
add_test(NAME omlx_m5_p04_tf_equality COMMAND $<TARGET_FILE:omlx_m5_tf_target_equality_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")

add_executable(omlx_m5_p05_page_pool_tests tests/p05_page_pool_tests.cpp)
target_link_libraries(omlx_m5_p05_page_pool_tests PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_p05_page_pool_tests omlx_m5_attention_postprocess_metallib)
add_test(NAME omlx_m5_p05_page_pool COMMAND $<TARGET_FILE:omlx_m5_p05_page_pool_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")

add_executable(omlx_m5_tf_attention_exact_tests tests/tf_attention_exact_tests.cpp)
target_include_directories(omlx_m5_tf_attention_exact_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_tf_attention_exact_tests PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_tf_attention_exact_tests omlx_m5_attention_exact_metallib)
add_test(NAME omlx_m5_p06_tf_attention_exact COMMAND $<TARGET_FILE:omlx_m5_tf_attention_exact_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")

add_executable(omlx_m5_benchmark_p06_qk_fusion apps/benchmark_qk_fusion.cpp)
target_include_directories(omlx_m5_benchmark_p06_qk_fusion PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_benchmark_p06_qk_fusion PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_benchmark_p06_qk_fusion
  omlx_m5_attention_postprocess_metallib)

add_executable(omlx_m5_benchmark_dflash_slice apps/benchmark_dflash_slice.cpp)
target_include_directories(omlx_m5_benchmark_dflash_slice PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_benchmark_dflash_slice PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_benchmark_dflash_slice omlx_m5_qwen_forward_resources)

add_executable(omlx_m5_benchmark_projection_candidates apps/benchmark_projection_candidates.cpp)
target_include_directories(omlx_m5_benchmark_projection_candidates PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_benchmark_projection_candidates PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_benchmark_projection_candidates omlx_m5_qwen_forward_resources)
add_dependencies(omlx_m5_benchmark_projection_candidates omlx_m5_quantized_matrix_metallib)

add_executable(omlx_m5_inspect_forward_routes apps/inspect_forward_routes.cpp)
target_include_directories(omlx_m5_inspect_forward_routes PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_inspect_forward_routes PRIVATE omlx_m5_runtime)

add_executable(omlx_m5_benchmark_gdn_prefill apps/benchmark_gdn_prefill.cpp)
target_include_directories(omlx_m5_benchmark_gdn_prefill PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_benchmark_gdn_prefill PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_benchmark_gdn_prefill omlx_m5_qwen_forward_resources
  omlx_m5_gdn_candidate_io_metallib omlx_m5_gdn_prefill_candidates_metallib)

add_executable(omlx_m5_benchmark_long_kv_attention apps/benchmark_long_kv_attention.cpp)
target_include_directories(omlx_m5_benchmark_long_kv_attention PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_benchmark_long_kv_attention PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_benchmark_long_kv_attention omlx_m5_qwen_forward_resources)
add_dependencies(omlx_m5_benchmark_long_kv_attention
  omlx_m5_attention_contiguous_control_metallib)
add_dependencies(omlx_m5_benchmark_long_kv_attention omlx_m5_attention_split_kv_metallib)

add_executable(omlx_m5_benchmark_p10_batch apps/benchmark_p10_batch.cpp)
target_include_directories(omlx_m5_benchmark_p10_batch PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_benchmark_p10_batch PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_benchmark_p10_batch omlx_m5_qwen_forward_resources)

if(TARGET omlx_m5_p11_fp4_native_metallib)
  add_executable(omlx_m5_p11_tensor_canary apps/p11_tensor_canary.mm)
  target_include_directories(omlx_m5_p11_tensor_canary PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p11_tensor_canary PRIVATE omlx_m5_runtime)
  target_compile_options(omlx_m5_p11_tensor_canary PRIVATE -fobjc-arc)
  add_dependencies(omlx_m5_p11_tensor_canary omlx_m5_p11_fp4_native_metallib)
  add_test(NAME omlx_m5_p11_tensor_canary COMMAND $<TARGET_FILE:omlx_m5_p11_tensor_canary>
    "${OMLX_M5_METAL_OUTPUT_DIR}/omlx_m5_p11_fp4_native.metallib")
endif()

if(TARGET omlx_m5_affine_lane_metallib)
  add_executable(omlx_m5_tf_affine_lane_tests tests/tf_affine_lane_tests.cpp)
  target_include_directories(omlx_m5_tf_affine_lane_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_affine_lane_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_affine_lane_tests omlx_m5_affine_lane_metallib)
  add_test(NAME omlx_m5_p11_tf_affine_lane COMMAND $<TARGET_FILE:omlx_m5_tf_affine_lane_tests>
    "${OMLX_M5_METAL_OUTPUT_DIR}")
endif()

add_executable(omlx_m5_p11_layout_inventory apps/p11_layout_inventory.cpp)
target_include_directories(omlx_m5_p11_layout_inventory PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_p11_layout_inventory PRIVATE omlx_m5_runtime)

add_executable(omlx_m5_export_p12_routes apps/export_p12_routes.cpp)
target_include_directories(omlx_m5_export_p12_routes PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(omlx_m5_export_p12_routes PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_export_p12_routes omlx_m5_qwen_forward_resources)

# Reconfigure when execution sources change, then bind the compiled runtime to
# the same source identity used by checkpoint and route provenance.
file(GLOB_RECURSE OMLX_M5_EXECUTION_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.in" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*")
list(APPEND OMLX_M5_EXECUTION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
list(SORT OMLX_M5_EXECUTION_SOURCES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${OMLX_M5_EXECUTION_SOURCES})
set(OMLX_M5_EXECUTION_HASH_INPUT "${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}:${CMAKE_BUILD_TYPE}:${CMAKE_CXX_FLAGS}")
foreach(OMLX_M5_EXECUTION_SOURCE IN LISTS OMLX_M5_EXECUTION_SOURCES)
  file(SHA256 "${OMLX_M5_EXECUTION_SOURCE}" OMLX_M5_EXECUTION_FILE_HASH)
  file(RELATIVE_PATH OMLX_M5_EXECUTION_RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${OMLX_M5_EXECUTION_SOURCE}")
  string(APPEND OMLX_M5_EXECUTION_HASH_INPUT "\n${OMLX_M5_EXECUTION_RELATIVE}:${OMLX_M5_EXECUTION_FILE_HASH}")
endforeach()
string(SHA256 OMLX_M5_EXECUTION_SOURCE_ID "${OMLX_M5_EXECUTION_HASH_INPUT}")
configure_file(src/execution_build_identity.h.in generated/execution_build_identity.h @ONLY)
target_include_directories(omlx_m5_core_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_executable(omlx_m5_step2_wide_nax_tests tests/step2_wide_nax_tests.cpp)
target_include_directories(omlx_m5_step2_wide_nax_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_options(omlx_m5_step2_wide_nax_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(omlx_m5_step2_wide_nax_tests PRIVATE omlx_m5_core_runtime)
add_dependencies(omlx_m5_step2_wide_nax_tests omlx_m5_qwen_forward_resources)
if(TARGET omlx_m5_mxfp4_nax_metallib)
  add_dependencies(omlx_m5_step2_wide_nax_tests
    omlx_m5_mxfp4_nax_metallib
    omlx_m5_affine_nax_metallib
    omlx_m5_bf16_nax_metallib)
endif()
add_test(NAME omlx_m5_step2_wide_nax COMMAND $<TARGET_FILE:omlx_m5_step2_wide_nax_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")

add_executable(omlx_m5_lora_dual_mode_tests tests/lora_dual_mode_tests.cpp)
target_include_directories(omlx_m5_lora_dual_mode_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_options(omlx_m5_lora_dual_mode_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(omlx_m5_lora_dual_mode_tests PRIVATE omlx_m5_runtime)
add_dependencies(omlx_m5_lora_dual_mode_tests omlx_m5_qwen_forward_resources)
add_test(NAME omlx_m5_lora_dual_mode COMMAND $<TARGET_FILE:omlx_m5_lora_dual_mode_tests>
  "${OMLX_M5_METAL_OUTPUT_DIR}")

if(TARGET omlx_m5_attention_nax_prefill_metallib AND TARGET omlx_m5_attention_affine4_metallib)
  add_executable(omlx_m5_m2_gqa_flash_tests tests/m2_gqa_flash_tests.cpp)
  target_include_directories(omlx_m5_m2_gqa_flash_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(omlx_m5_m2_gqa_flash_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(omlx_m5_m2_gqa_flash_tests PRIVATE omlx_m5_core_runtime)
  add_dependencies(omlx_m5_m2_gqa_flash_tests
    omlx_m5_attention_nax_prefill_metallib
    omlx_m5_attention_affine4_metallib)
  add_test(NAME omlx_m5_m2_gqa_flash COMMAND $<TARGET_FILE:omlx_m5_m2_gqa_flash_tests>
    "${OMLX_M5_METAL_OUTPUT_DIR}")
endif()

if(TARGET omlx_m5_attention_sage2_metallib AND TARGET omlx_m5_attention_minference_metallib)
  add_executable(omlx_m5_m2_sage2_minference_tests tests/m2_sage2_minference_tests.cpp)
  target_include_directories(omlx_m5_m2_sage2_minference_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(omlx_m5_m2_sage2_minference_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(omlx_m5_m2_sage2_minference_tests PRIVATE omlx_m5_core_runtime)
  add_dependencies(omlx_m5_m2_sage2_minference_tests
    omlx_m5_attention_nax_prefill_metallib
    omlx_m5_attention_affine4_metallib
    omlx_m5_attention_sage2_metallib
    omlx_m5_attention_minference_metallib)
  add_test(NAME omlx_m5_m2_sage2_minference COMMAND $<TARGET_FILE:omlx_m5_m2_sage2_minference_tests>
    "${OMLX_M5_METAL_OUTPUT_DIR}")
endif()

if(OMLX_M5_BUILD_MLX_FIXTURES)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution/fixture.stamp"
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_qwen_execution_fixture.py"
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_qwen_execution_fixture.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qwen_reference.py" "${CMAKE_CURRENT_SOURCE_DIR}/tools/m5_artifact.py"
    VERBATIM)
  add_custom_target(omlx_m5_qwen_execution_fixture
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution/fixture.stamp")

  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution/fixture.stamp"
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_dflash_execution_fixture.py"
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution"
      --target "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution/affine"
    DEPENDS omlx_m5_qwen_execution_fixture "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_dflash_execution_fixture.py"
    VERBATIM)
  add_custom_target(omlx_m5_dflash_execution_fixture
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution/fixture.stamp")

  add_dependencies(omlx_m5_b6_native_backend_tests
    omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture
    omlx_m5_dflash_execution_fixture)
  if(TARGET omlx_m5_native)
    add_dependencies(omlx_m5_native
      omlx_m5_qwen_execution_fixture
      omlx_m5_dflash_execution_fixture)
  endif()

  add_executable(omlx_m5_qwen_forward_tests tests/qwen_forward_tests.cpp)
  target_include_directories(omlx_m5_qwen_forward_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_qwen_forward_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_qwen_forward_tests
    omlx_m5_moe_metallib
    omlx_m5_qwen_forward_metallib omlx_m5_vocab_metallib omlx_m5_qmv_t1_metallib
    omlx_m5_qmm_small_t_metallib omlx_m5_qmm_broad_metallib
    omlx_m5_mxfp4_small_t_metallib omlx_m5_rmsnorm_projection_metallib
    omlx_m5_attention_postprocess_metallib omlx_m5_gdn_layer_metallib omlx_m5_mlp_metallib
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_qwen_forward COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_qwen_execution.py" $<TARGET_FILE:omlx_m5_qwen_forward_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")
  set_tests_properties(omlx_m5_qwen_forward PROPERTIES RESOURCE_LOCK qwen_execution_fixture)

  add_executable(omlx_m5_p02_route_tests tests/p02_route_tests.cpp)
  target_include_directories(omlx_m5_p02_route_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p02_route_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p02_route_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p02_route_ab COMMAND $<TARGET_FILE:omlx_m5_p02_route_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_dflash_forward_tests tests/dflash_forward_tests.cpp)
  target_include_directories(omlx_m5_dflash_forward_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_dflash_forward_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_dflash_forward_tests omlx_m5_qwen_forward_resources omlx_m5_dflash_execution_fixture)
  add_test(NAME omlx_m5_dflash_forward COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate_dflash_execution.py" $<TARGET_FILE:omlx_m5_dflash_forward_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")
  add_test(NAME omlx_m5_p02_tf_draft_linear COMMAND $<TARGET_FILE:omlx_m5_dflash_forward_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" "${OMLX_M5_METAL_OUTPUT_DIR}" --row-exact)
  add_test(NAME omlx_m5_p03_tf_draft_position COMMAND $<TARGET_FILE:omlx_m5_dflash_forward_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" "${OMLX_M5_METAL_OUTPUT_DIR}" --row-exact --position-coupled)
  set_tests_properties(
    omlx_m5_dflash_forward
    omlx_m5_p02_tf_draft_linear
    omlx_m5_p03_tf_draft_position
    PROPERTIES
    RESOURCE_LOCK dflash_execution_fixture)

  add_executable(omlx_m5_tf_coupled_engine_tests tests/tf_coupled_engine_tests.cpp)
  target_include_directories(omlx_m5_tf_coupled_engine_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_coupled_engine_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_coupled_engine_tests omlx_m5_qwen_forward_resources omlx_m5_dflash_execution_fixture)
  add_test(NAME omlx_m5_p04_tf_engine COMMAND $<TARGET_FILE:omlx_m5_tf_coupled_engine_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_p04_verify_tests tests/p04_target_verify_tests.cpp)
  target_include_directories(omlx_m5_p04_verify_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p04_verify_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p04_verify_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p04_verify COMMAND $<TARGET_FILE:omlx_m5_p04_verify_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_p05_qwen_paging_tests tests/p05_qwen_paging_tests.cpp)
  target_include_directories(omlx_m5_p05_qwen_paging_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p05_qwen_paging_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p05_qwen_paging_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p05_qwen_paging COMMAND $<TARGET_FILE:omlx_m5_p05_qwen_paging_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_tf_gdn_state_tests tests/tf_gdn_state_tests.cpp)
  target_include_directories(omlx_m5_tf_gdn_state_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_gdn_state_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_gdn_state_tests omlx_m5_qwen_forward_resources omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p07_tf_state COMMAND $<TARGET_FILE:omlx_m5_tf_gdn_state_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_p06_attention_tests tests/p06_attention_tests.cpp)
  target_include_directories(omlx_m5_p06_attention_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p06_attention_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p06_attention_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p06_attention COMMAND $<TARGET_FILE:omlx_m5_p06_attention_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")
  add_test(NAME omlx_m5_p06_tf_forward_exact COMMAND $<TARGET_FILE:omlx_m5_p06_attention_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}" --row-exact)

  add_executable(omlx_m5_p08_prefix_tests tests/p08_prefix_tests.cpp)
  target_include_directories(omlx_m5_p08_prefix_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p08_prefix_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p08_prefix_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture omlx_m5_dflash_execution_fixture)
  add_test(NAME omlx_m5_p08_prefix COMMAND $<TARGET_FILE:omlx_m5_p08_prefix_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution")

  add_executable(omlx_m5_p09_concurrency_tests tests/p09_concurrency_tests.cpp)
  target_include_directories(omlx_m5_p09_concurrency_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p09_concurrency_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p09_concurrency_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture omlx_m5_dflash_execution_fixture)
  add_test(NAME omlx_m5_p09_concurrency COMMAND $<TARGET_FILE:omlx_m5_p09_concurrency_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution")
  add_test(NAME omlx_m5_p10_batch COMMAND $<TARGET_FILE:omlx_m5_p09_concurrency_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" --p10)

  add_executable(omlx_m5_p10_batch_state_tests tests/p10_batch_state_tests.cpp)
  target_include_directories(omlx_m5_p10_batch_state_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p10_batch_state_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p10_batch_state_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p10_batch_state COMMAND $<TARGET_FILE:omlx_m5_p10_batch_state_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_p11_tensor_view_tests tests/p11_tensor_view_tests.cpp)
  target_link_libraries(omlx_m5_p11_tensor_view_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p11_tensor_view_tests omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p11_tensor_view COMMAND $<TARGET_FILE:omlx_m5_p11_tensor_view_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution/mxfp4/model.omlxm5")

  add_executable(omlx_m5_p12_route_tests tests/p12_route_tests.cpp)
  target_include_directories(omlx_m5_p12_route_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_p12_route_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_p12_route_tests omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p12_route COMMAND $<TARGET_FILE:omlx_m5_p12_route_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution/moe-mxfp4"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/p12-control-trace.json")
  add_test(NAME omlx_m5_p12_route_affine COMMAND $<TARGET_FILE:omlx_m5_p12_route_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution/moe-affine"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/p12-control-affine.json")

  add_executable(omlx_m5_tf_execution_identity_tests tests/tf_execution_identity_tests.cpp)
  target_link_libraries(omlx_m5_tf_execution_identity_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_execution_identity_tests omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_tf_execution_identity COMMAND $<TARGET_FILE:omlx_m5_tf_execution_identity_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution")

  add_executable(omlx_m5_tf_coupled_state_tests tests/tf_coupled_state_tests.cpp)
  target_include_directories(omlx_m5_tf_coupled_state_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_coupled_state_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_coupled_state_tests omlx_m5_qwen_forward_resources omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p04_tf_state COMMAND $<TARGET_FILE:omlx_m5_tf_coupled_state_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_tf_prefix_contract_tests tests/tf_prefix_contract_tests.cpp)
  target_include_directories(omlx_m5_tf_prefix_contract_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_prefix_contract_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_prefix_contract_tests omlx_m5_qwen_forward_resources omlx_m5_qwen_execution_fixture omlx_m5_dflash_execution_fixture)
  add_test(NAME omlx_m5_p08_tf_prefix COMMAND $<TARGET_FILE:omlx_m5_tf_prefix_contract_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_tf_coupled_slots_tests tests/tf_coupled_slots_tests.cpp)
  target_include_directories(omlx_m5_tf_coupled_slots_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_coupled_slots_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_coupled_slots_tests omlx_m5_qwen_forward_resources omlx_m5_qwen_execution_fixture omlx_m5_dflash_execution_fixture)
  add_test(NAME omlx_m5_p09_tf_slots COMMAND $<TARGET_FILE:omlx_m5_tf_coupled_slots_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")
  add_test(NAME omlx_m5_p10_tf_native COMMAND $<TARGET_FILE:omlx_m5_tf_coupled_slots_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution" "${OMLX_M5_METAL_OUTPUT_DIR}" --batch-prototype)

  add_executable(omlx_m5_tf_batch_projection_tests tests/tf_batch_projection_tests.cpp)
  target_include_directories(omlx_m5_tf_batch_projection_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_batch_projection_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_batch_projection_tests omlx_m5_qwen_forward_resources omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p10_tf_projection COMMAND $<TARGET_FILE:omlx_m5_tf_batch_projection_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_tf_route_contract_tests tests/tf_route_contract_tests.cpp)
  target_include_directories(omlx_m5_tf_route_contract_tests PRIVATE ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_link_libraries(omlx_m5_tf_route_contract_tests PRIVATE omlx_m5_runtime)
  add_dependencies(omlx_m5_tf_route_contract_tests
    omlx_m5_qwen_forward_resources
    omlx_m5_export_p12_routes
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_p12_tf_contract COMMAND "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/tf_route_contract_ctest.py"
    "$<TARGET_FILE:omlx_m5_export_p12_routes>" "$<TARGET_FILE:omlx_m5_tf_route_contract_tests>"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution" "${OMLX_M5_METAL_OUTPUT_DIR}")

  add_executable(omlx_m5_w2_ninfer_features_tests tests/w2_ninfer_features_tests.cpp)
  target_include_directories(omlx_m5_w2_ninfer_features_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(omlx_m5_w2_ninfer_features_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(omlx_m5_w2_ninfer_features_tests PRIVATE omlx_m5_core_runtime)
  add_dependencies(omlx_m5_w2_ninfer_features_tests
    omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture
    omlx_m5_dflash_execution_fixture)
  add_test(NAME omlx_m5_w2_ninfer_features COMMAND $<TARGET_FILE:omlx_m5_w2_ninfer_features_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/dflash-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}")

  if(TARGET omlx_m5_gdn_chunk_nax_metallib)
    add_executable(omlx_m5_gdn_chunk_nax_tests tests/gdn_chunk_nax_tests.cpp)
    target_include_directories(omlx_m5_gdn_chunk_nax_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
    target_compile_options(omlx_m5_gdn_chunk_nax_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(omlx_m5_gdn_chunk_nax_tests PRIVATE omlx_m5_core_runtime)
    add_dependencies(omlx_m5_gdn_chunk_nax_tests
      omlx_m5_qwen_forward_resources
      omlx_m5_qwen_execution_fixture)
    add_test(NAME omlx_m5_gdn_chunk_nax COMMAND $<TARGET_FILE:omlx_m5_gdn_chunk_nax_tests>
      "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
      "${OMLX_M5_METAL_OUTPUT_DIR}")
  endif()

  add_executable(omlx_m5_mtp_forward_tests tests/mtp_forward_tests.cpp)
  target_include_directories(omlx_m5_mtp_forward_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(omlx_m5_mtp_forward_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(omlx_m5_mtp_forward_tests PRIVATE omlx_m5_core_runtime)
  add_dependencies(omlx_m5_mtp_forward_tests
    omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  add_test(NAME omlx_m5_step5_mtp_forward COMMAND $<TARGET_FILE:omlx_m5_mtp_forward_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}")

  add_executable(omlx_m5_step4_moe_ar_tests tests/step4_moe_ar_tests.cpp)
  target_include_directories(omlx_m5_step4_moe_ar_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(omlx_m5_step4_moe_ar_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(omlx_m5_step4_moe_ar_tests PRIVATE omlx_m5_core_runtime)
  add_dependencies(omlx_m5_step4_moe_ar_tests
    omlx_m5_qwen_forward_resources
    omlx_m5_qwen_execution_fixture)
  if(TARGET omlx_m5_moe_nax_metallib)
    add_dependencies(omlx_m5_step4_moe_ar_tests omlx_m5_moe_nax_metallib)
  endif()
  if(TARGET omlx_m5_mxfp4_nax_metallib)
    add_dependencies(omlx_m5_step4_moe_ar_tests omlx_m5_mxfp4_nax_metallib)
  endif()
  if(TARGET omlx_m5_affine_nax_metallib)
    add_dependencies(omlx_m5_step4_moe_ar_tests omlx_m5_affine_nax_metallib)
  endif()
  add_test(NAME omlx_m5_step4_moe_ar COMMAND $<TARGET_FILE:omlx_m5_step4_moe_ar_tests>
    "${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen-execution"
    "${OMLX_M5_METAL_OUTPUT_DIR}"
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Native C++ standalone OpenAI-compatible HTTP server (ninfer-metal-serve).
add_library(
  omlx_m5_utf8proc
  STATIC
  third_party/utf8proc/utf8proc.c)
target_include_directories(
  omlx_m5_utf8proc
  SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(
  omlx_m5_jinja
  STATIC
  third_party/llama-jinja/jinja/json.cpp
  third_party/llama-jinja/jinja/lexer.cpp
  third_party/llama-jinja/jinja/parser.cpp
  third_party/llama-jinja/jinja/runtime.cpp
  third_party/llama-jinja/jinja/string.cpp
  third_party/llama-jinja/jinja/unicode.cpp
  third_party/llama-jinja/jinja/value.cpp)
target_compile_features(omlx_m5_jinja PUBLIC cxx_std_20)
target_include_directories(
  omlx_m5_jinja
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(
  omlx_m5_jinja
  SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-jinja
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_link_libraries(
  omlx_m5_jinja
  PUBLIC
  omlx_m5_utf8proc)

add_library(
  omlx_m5_serve_lib
  STATIC
  src/serve/chat_template.cpp
  src/serve/config.cpp
  src/serve/engine_router.cpp
  src/serve/http_server.cpp
  src/serve/jinja.cpp
  src/serve/multi_model_host.cpp
  src/serve/output_parser.cpp
  src/serve/provider_registry.cpp
  src/serve/tokenizer.cpp
  src/serve/uds_sidecar_backend.cpp
  src/serve/unicode.cpp)
target_compile_features(omlx_m5_serve_lib PUBLIC cxx_std_20)
target_include_directories(
  omlx_m5_serve_lib
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(
  omlx_m5_serve_lib
  SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-jinja
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_definitions(
  omlx_m5_serve_lib
  PUBLIC
  NINFER_METAL_DISPLAY_VERSION="${NINFER_METAL_DISPLAY_VERSION}"
  NINFER_METAL_CODENAME="${NINFER_METAL_CODENAME}"
  NINFER_METAL_SEMVER="${NINFER_METAL_SEMVER}"
  NINFER_METAL_CORE_ABI=${NINFER_METAL_CORE_ABI}
  PRIVATE
  OMLX_M5_DEFAULT_METALLIB_DIR="${OMLX_M5_METAL_OUTPUT_DIR}")
target_compile_options(
  omlx_m5_serve_lib
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)
target_link_libraries(
  omlx_m5_serve_lib
  PUBLIC
  omlx_m5_core_runtime
  PRIVATE
  omlx_m5_jinja
  omlx_m5_utf8proc)

add_executable(
  ninfer-metal-serve
  apps/ninfer_metal_serve.cpp)
target_compile_definitions(
  ninfer-metal-serve
  PRIVATE
  NINFER_METAL_DISPLAY_VERSION="${NINFER_METAL_DISPLAY_VERSION}"
  NINFER_METAL_CODENAME="${NINFER_METAL_CODENAME}"
  NINFER_METAL_SEMVER="${NINFER_METAL_SEMVER}"
  NINFER_METAL_CORE_ABI=${NINFER_METAL_CORE_ABI})
target_compile_options(
  ninfer-metal-serve
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)
target_link_libraries(
  ninfer-metal-serve
  PRIVATE
  omlx_m5_serve_lib
  omlx_m5_core_runtime)
add_dependencies(
  ninfer-metal-serve
  omlx_m5_qwen_forward_resources)

add_executable(
  omlx_m5_mock_spi_provider
  tests/mock_spi_provider.cpp)
target_compile_features(omlx_m5_mock_spi_provider PRIVATE cxx_std_20)
target_include_directories(
  omlx_m5_mock_spi_provider
  SYSTEM PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_options(
  omlx_m5_mock_spi_provider
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)

add_executable(
  omlx_m5_serve_e2e
  tests/serve_e2e_tests.cpp)
target_include_directories(
  omlx_m5_serve_e2e
  SYSTEM PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-jinja
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_options(
  omlx_m5_serve_e2e
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)
target_link_libraries(
  omlx_m5_serve_e2e
  PRIVATE
  omlx_m5_serve_lib
  omlx_m5_core_runtime)
add_dependencies(
  omlx_m5_serve_e2e
  omlx_m5_mock_spi_provider)
add_test(
  NAME omlx_m5_serve_e2e
  COMMAND
    $<TARGET_FILE:omlx_m5_serve_e2e>
    "${CMAKE_SOURCE_DIR}/models"
    $<TARGET_FILE:omlx_m5_mock_spi_provider>)

add_executable(
  omlx_m5_prefill_switches_spi_tests
  tests/prefill_switches_spi_tests.cpp)
target_include_directories(
  omlx_m5_prefill_switches_spi_tests
  SYSTEM PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_options(
  omlx_m5_prefill_switches_spi_tests
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)
target_link_libraries(
  omlx_m5_prefill_switches_spi_tests
  PRIVATE
  omlx_m5_serve_lib
  omlx_m5_core_runtime)
add_dependencies(
  omlx_m5_prefill_switches_spi_tests
  mseries_backendd)
add_test(
  NAME prefill_switches_spi_tests
  COMMAND
    $<TARGET_FILE:omlx_m5_prefill_switches_spi_tests>
    $<TARGET_FILE:mseries_backendd>)

# Module 1: Route A W4A8 Dense & MoE Metal 4 NAX kernels & standalone verification tests.
if(OMLX_M5_BUILD_NAX_METALLIB AND
   OMLX_M5_MACOS_SDK_VERSION VERSION_GREATER_EQUAL 27.0)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/w4a8_nax.metal")
    omlx_m5_compile_metallib(
      omlx_m5_w4a8_nax_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/w4a8_nax.metal"
      "omlx_m5_w4a8_nax.metallib"
      "27.0"
      EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_w4a8_nax_metallib)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/w4a8_moe_nax.metal")
    omlx_m5_compile_metallib(
      omlx_m5_w4a8_moe_nax_metallib
      "${CMAKE_CURRENT_SOURCE_DIR}/kernels/w4a8_moe_nax.metal"
      "omlx_m5_w4a8_moe_nax.metallib"
      "27.0"
      EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
    add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_w4a8_moe_nax_metallib)
  endif()
  if(TARGET omlx_m5_w4a8_nax_metallib AND TARGET omlx_m5_w4a8_moe_nax_metallib)
    add_executable(omlx_m5_w4a8_nax_tests tests/w4a8_nax_tests.cpp)
    target_include_directories(omlx_m5_w4a8_nax_tests PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
    target_compile_options(omlx_m5_w4a8_nax_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(omlx_m5_w4a8_nax_tests PRIVATE omlx_m5_core_runtime)
    add_dependencies(omlx_m5_w4a8_nax_tests
      omlx_m5_qwen_forward_resources
      omlx_m5_w4a8_nax_metallib
      omlx_m5_w4a8_moe_nax_metallib
      omlx_m5_affine_nax_metallib
      omlx_m5_mxfp4_nax_metallib
      omlx_m5_moe_nax_metallib)
    if(TARGET omlx_m5_gdn_chunk_nax_metallib)
      add_dependencies(omlx_m5_w4a8_nax_tests omlx_m5_gdn_chunk_nax_metallib)
    endif()
    add_test(NAME omlx_m5_w4a8_nax COMMAND $<TARGET_FILE:omlx_m5_w4a8_nax_tests>
      "${OMLX_M5_METAL_OUTPUT_DIR}")
  endif()
endif()

# Module 4b: Native Dual-ANE Fixed-Chunk Output-Channel-Split Co-Execution Engine
omlx_m5_compile_metallib(
  omlx_m5_ane_pack_unpack_metallib
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ane_pack_unpack.metal"
  "omlx_m5_ane_pack_unpack.metallib"
  "15.0")
add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_ane_pack_unpack_metallib)

target_sources(
  omlx_m5_core_runtime
  PRIVATE
  src/ane/ane_dual_offload.mm)
target_include_directories(
  omlx_m5_core_runtime
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(omlx_m5_ane_dual_offload INTERFACE)
target_include_directories(
  omlx_m5_ane_dual_offload
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
  omlx_m5_ane_dual_offload
  INTERFACE
  omlx_m5_core_runtime)

add_executable(
  omlx_m5_ane_dual_offload_tests
  tests/ane_dual_offload_tests.mm)
target_include_directories(
  omlx_m5_ane_dual_offload_tests
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_features(omlx_m5_ane_dual_offload_tests PRIVATE cxx_std_20)
target_compile_options(
  omlx_m5_ane_dual_offload_tests
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(
  omlx_m5_ane_dual_offload_tests
  PRIVATE
  omlx_m5_ane_dual_offload
  omlx_m5_core_runtime)
add_dependencies(
  omlx_m5_ane_dual_offload_tests
  omlx_m5_ane_pack_unpack_metallib)
add_test(
  NAME omlx_m5_ane_dual_offload
  COMMAND
    $<TARGET_FILE:omlx_m5_ane_dual_offload_tests>
    "${OMLX_M5_METAL_OUTPUT_DIR}")

# Module 3: Hybrid 48-GDN + 16-Full-Attn Cross-Chunk Reuse (`GDN-CacheBlend`).
target_sources(
  omlx_m5_core_runtime
  PRIVATE
  src/chunk_blend_cache.cpp)

if(OMLX_M5_BUILD_NAX_METALLIB AND
   OMLX_M5_MACOS_SDK_VERSION VERSION_GREATER_EQUAL 27.0)
  omlx_m5_compile_metallib(
    omlx_m5_cache_blend_metallib
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cache_blend.metal"
    "omlx_m5_cache_blend.metallib"
    "27.0"
    EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common_math.h")
  add_dependencies(omlx_m5_qwen_forward_resources omlx_m5_cache_blend_metallib)

  add_executable(
    omlx_m5_cache_blend_hybrid_tests
    tests/cache_blend_hybrid_tests.cpp)
  target_include_directories(
    omlx_m5_cache_blend_hybrid_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(
    omlx_m5_cache_blend_hybrid_tests
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  target_link_libraries(
    omlx_m5_cache_blend_hybrid_tests
    PRIVATE
    omlx_m5_core_runtime)
  add_dependencies(
    omlx_m5_cache_blend_hybrid_tests
    omlx_m5_cache_blend_metallib
    omlx_m5_gdn_chunk_nax_metallib
    omlx_m5_gdn_layer_metallib
    omlx_m5_attention_affine4_metallib
    omlx_m5_attention_postprocess_metallib)
  add_test(
    NAME omlx_m5_cache_blend_hybrid
    COMMAND
      $<TARGET_FILE:omlx_m5_cache_blend_hybrid_tests>
      "${OMLX_M5_METAL_OUTPUT_DIR}")
endif()

# Module 5: Full-Matrix Prefill / PPL / Logits Benchmark Helper
add_executable(
  omlx_m5_prefill_matrix_bench
  apps/prefill_matrix_bench.cpp)
target_include_directories(
  omlx_m5_prefill_matrix_bench
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(
  omlx_m5_prefill_matrix_bench
  SYSTEM PRIVATE
  ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
target_compile_options(
  omlx_m5_prefill_matrix_bench
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror)
target_link_libraries(
  omlx_m5_prefill_matrix_bench
  PRIVATE
  omlx_m5_serve_lib
  omlx_m5_core_runtime)
add_dependencies(
  omlx_m5_prefill_matrix_bench
  omlx_m5_qwen_forward_resources)

# Wave 2: Unified End-to-End All-Modes Prefill Integration Test
if(OMLX_M5_BUILD_NAX_METALLIB AND
   OMLX_M5_MACOS_SDK_VERSION VERSION_GREATER_EQUAL 27.0)
  add_executable(
    omlx_m5_prefill_all_modes_e2e_tests
    tests/prefill_all_modes_e2e_tests.cpp)
  target_include_directories(
    omlx_m5_prefill_all_modes_e2e_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
  target_compile_options(
    omlx_m5_prefill_all_modes_e2e_tests
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror)
  target_link_libraries(
    omlx_m5_prefill_all_modes_e2e_tests
    PRIVATE
    omlx_m5_core_runtime)
  add_dependencies(
    omlx_m5_prefill_all_modes_e2e_tests
    omlx_m5_qwen_forward_resources)
  add_test(
    NAME omlx_m5_prefill_all_modes_e2e
    COMMAND
      $<TARGET_FILE:omlx_m5_prefill_all_modes_e2e_tests>
      "${OMLX_M5_METAL_OUTPUT_DIR}")

  # WP3 Global Apple10 <= 16.0 KiB Static TG SRAM & 0-Byte Register-Spill CI Audit Gate
  if(TARGET omlx_m5_attention_nax_prefill_metallib AND
     TARGET omlx_m5_attention_affine4_metallib AND
     TARGET omlx_m5_gdn_chunk_nax_metallib AND
     TARGET omlx_m5_cache_blend_metallib)
    add_executable(
      omlx_m5_prefill_sram_spill_audit_tests
      tests/prefill_sram_spill_audit_tests.mm)
    target_include_directories(
      omlx_m5_prefill_sram_spill_audit_tests
      PRIVATE
      ${OMLX_M5_THIRD_PARTY_INCLUDE_DIR})
    target_compile_features(omlx_m5_prefill_sram_spill_audit_tests PRIVATE cxx_std_20)
    target_compile_options(
      omlx_m5_prefill_sram_spill_audit_tests
      PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
    target_link_libraries(
      omlx_m5_prefill_sram_spill_audit_tests
      PRIVATE
      "-framework Foundation"
      "-framework Metal")
    add_dependencies(
      omlx_m5_prefill_sram_spill_audit_tests
      omlx_m5_attention_nax_prefill_metallib
      omlx_m5_attention_affine4_metallib
      omlx_m5_gdn_chunk_nax_metallib
      omlx_m5_cache_blend_metallib)
    add_test(
      NAME omlx_m5_prefill_sram_spill_audit
      COMMAND
        $<TARGET_FILE:omlx_m5_prefill_sram_spill_audit_tests>
        "${OMLX_M5_METAL_OUTPUT_DIR}")
  endif()
endif()
