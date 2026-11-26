cmake_minimum_required(VERSION 3.20)

set(IMTALKER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Directory containing IMTalker LiteRT C++ exporter sources")

function(add_imtalker_exporter target_name src_file)
  add_executable(${target_name} EXCLUDE_FROM_ALL
      ../utils/minijson.cc
      ../utils/safetensor_loader.cc
      ../utils/safetensors.cc
      "${IMTALKER_SRC_DIR}/${src_file}"
  )

  target_include_directories(${target_name}
      PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
          "${CMAKE_CURRENT_SOURCE_DIR}/../.."
          "${TFLITE_SOURCE_DIR}"
  )

  target_link_libraries(${target_name}
      PRIVATE
          litert_tensor_api_core
          litert_tensor_api_tflite_backend
          tensorflow-lite
          absl::flat_hash_map
          absl::flat_hash_set
          absl::log
          absl::status
          absl::statusor
          absl::str_format
          absl::strings
          absl::string_view
          absl::span
  )
endfunction()

add_imtalker_exporter(imtalker_fm_export "fm_step_export.cc")
add_imtalker_exporter(imtalker_renderer_export "renderer_decode_export.cc")
add_imtalker_exporter(imtalker_wav2vec2_export "wav2vec2_export.cc")
add_imtalker_exporter(imtalker_student_export "student_renderer_export.cc")
