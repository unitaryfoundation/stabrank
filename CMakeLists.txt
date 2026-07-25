cmake_minimum_required(VERSION 3.21)
project(stabrank LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# scikit-build-core wheel builds set Release themselves, but a bare
# `cmake -S . -B build` (the test/benchmark path) otherwise compiles
# with no optimization at all, which badly misrepresents the SA kernels.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Opt-in host-tuned codegen (AVX2/FMA etc. for Eigen's complex kernels).
# Keep OFF for distributable wheels; turn ON for local source builds:
#   cmake -DSTABRANK_NATIVE=ON ...
#   SKBUILD_CMAKE_DEFINE="STABRANK_NATIVE=ON" uv pip install -e .
option(STABRANK_NATIVE "Compile with -march=native" OFF)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
endif()

# Core C++ library and tests
add_subdirectory(cpp)

# Python bindings (only when nanobind is available, i.e. via scikit-build-core)
if(NOT SKBUILD)
  # Standalone build: try to find nanobind, skip if not found
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
      RESULT_VARIABLE NB_DIR_RET)
    if(NB_DIR_RET EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    endif()
  endif()
endif()

if(SKBUILD)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
endif()

find_package(nanobind CONFIG QUIET)
if(nanobind_FOUND)
  add_subdirectory(python)
endif()
