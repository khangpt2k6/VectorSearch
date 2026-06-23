cmake_minimum_required(VERSION 3.16)
project(vectorsearch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Core library
add_library(vectorsearch
  src/flat_index.cpp
  src/hnsw_index.cpp
  src/scalar_quantizer.cpp
  src/scalar_quantized_index.cpp
)
target_include_directories(vectorsearch
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(vectorsearch PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(vectorsearch PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(vectorsearch PRIVATE /W4)
else()
  target_compile_options(vectorsearch PRIVATE -Wall -Wextra)
endif()

# Benchmarks (manual measurement programs, not CTest gates)
add_executable(recall_sweep benchmarks/recall_sweep.cpp)
target_link_libraries(recall_sweep PRIVATE vectorsearch)

# Tests
enable_testing()

add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance PRIVATE vectorsearch)
target_include_directories(test_distance PRIVATE tests)
add_test(NAME distance COMMAND test_distance)

add_executable(test_flat tests/test_flat.cpp)
target_link_libraries(test_flat PRIVATE vectorsearch)
target_include_directories(test_flat PRIVATE tests)
add_test(NAME flat COMMAND test_flat)

add_executable(test_hnsw tests/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE vectorsearch)
target_include_directories(test_hnsw PRIVATE tests)
add_test(NAME hnsw COMMAND test_hnsw)

add_executable(test_recall tests/test_recall.cpp)
target_link_libraries(test_recall PRIVATE vectorsearch)
target_include_directories(test_recall PRIVATE tests)
add_test(NAME recall COMMAND test_recall)

add_executable(test_scalar_quantizer tests/test_scalar_quantizer.cpp)
target_link_libraries(test_scalar_quantizer PRIVATE vectorsearch)
target_include_directories(test_scalar_quantizer PRIVATE tests)
add_test(NAME scalar_quantizer COMMAND test_scalar_quantizer)

add_executable(test_scalar_quantized_index tests/test_scalar_quantized_index.cpp)
target_link_libraries(test_scalar_quantized_index PRIVATE vectorsearch)
target_include_directories(test_scalar_quantized_index PRIVATE tests)
add_test(NAME scalar_quantized_index COMMAND test_scalar_quantized_index)
