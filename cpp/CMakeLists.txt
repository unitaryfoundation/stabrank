include(FetchContent)

# --- Eigen (header-only linear algebra) ---
# Resolution order:
#   1. an installed Eigen 3.4 (Ubuntu `libeigen3-dev` is 3.4.0; CI installs
#      it), found through its CMake package config;
#   2. the pinned 3.4.0 release tarball, downloaded from the GitHub mirror
#      first and from GitLab if the mirror fails. The two archives are
#      byte-identical, so a single SHA256 covers both and CMake moves on to
#      the next URL when a download fails or its hash does not match;
#   3. an already-populated source tree. A URL download has no update step,
#      so a reconfigure never touches the network once _deps/eigen-src
#      exists, and -DFETCHCONTENT_FULLY_DISCONNECTED=ON (or
#      -DFETCHCONTENT_SOURCE_DIR_EIGEN=<dir>) skips the download entirely.
# Keep STABRANK_EIGEN_VERSION and STABRANK_EIGEN_SHA256 in step: the hash is
# that of eigen-<version>.tar.gz as served by both hosts.
# Eigen's own CMakeLists registers every build tree in the user package
# registry (~/.cmake/packages), so without these two settings a stale
# _deps/eigen-build from some other checkout would satisfy find_package.
set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT Eigen3_FOUND)
  set(STABRANK_EIGEN_VERSION 3.4.0)
  set(STABRANK_EIGEN_SHA256
    8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
  if(POLICY CMP0135)
    # Extracted files take the extraction time as their timestamp.
    cmake_policy(SET CMP0135 NEW)
  endif()
  FetchContent_Declare(
    eigen
    URL
      https://github.com/eigen-mirror/eigen/archive/refs/tags/${STABRANK_EIGEN_VERSION}.tar.gz
      https://gitlab.com/libeigen/eigen/-/archive/${STABRANK_EIGEN_VERSION}/eigen-${STABRANK_EIGEN_VERSION}.tar.gz
    URL_HASH SHA256=${STABRANK_EIGEN_SHA256}
  )
  # Eigen's CMakeLists builds tests/docs by default; disable them without
  # permanently disabling this project's CTest registration.
  set(STABRANK_BUILD_TESTING "${BUILD_TESTING}")
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
  set(BUILD_TESTING "${STABRANK_BUILD_TESTING}" CACHE BOOL "" FORCE)
  message(STATUS "Eigen ${STABRANK_EIGEN_VERSION} from ${eigen_SOURCE_DIR}")
else()
  message(STATUS "Eigen ${Eigen3_VERSION} from ${Eigen3_DIR}")
endif()

# --- Catch2 (C++ testing) ---
if(BUILD_TESTING)
  find_package(Catch2 3 CONFIG QUIET)
  if(NOT Catch2_FOUND)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.7.1
      GIT_SHALLOW    TRUE
    )
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Catch2)
  endif()
endif()

# --- Core static library ---
add_library(stabrank_core_lib STATIC
  src/linalg.cpp
  src/polynomial.cpp
  src/pauli.cpp
  src/clifford.cpp
  src/fidelity.cpp
  src/sa_engine.cpp
  src/symmetric_engine.cpp
  src/pivot_pair.cpp
  src/t3_scan.cpp
  src/slice_match.cpp
  src/cover5.cpp
  src/cover6.cpp
  src/slice_match3.cpp
  src/dense_solve.cpp
)

target_include_directories(stabrank_core_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(Threads REQUIRED)
target_link_libraries(stabrank_core_lib PUBLIC Eigen3::Eigen Threads::Threads)

if(STABRANK_NATIVE AND NOT MSVC)
  target_compile_options(stabrank_core_lib PUBLIC -march=native)
endif()

if(BUILD_TESTING)
  # --- Catch2 tests ---
  add_executable(stabrank_tests
    tests/test_linalg.cpp
    tests/test_incremental_lsq.cpp
    tests/test_polynomial.cpp
    tests/test_pauli.cpp
    tests/test_sa_engine.cpp
    tests/test_symmetric_engine.cpp
    tests/test_kernel_properties.cpp
    tests/test_pivot_pair.cpp
    tests/test_t3_scan.cpp
    tests/test_slice_match.cpp
    tests/test_cover5.cpp
    tests/test_cover6.cpp
    tests/test_slice_match3.cpp
    tests/test_dense_solve.cpp
  )

  target_compile_definitions(stabrank_tests PRIVATE
    STABRANK_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )

  target_link_libraries(stabrank_tests PRIVATE
    stabrank_core_lib
    Catch2::Catch2WithMain
  )

  add_test(NAME stabrank_tests COMMAND stabrank_tests)

  # --- Catch2 micro-benchmarks (separate binary, not registered with ctest) ---
  add_executable(stabrank_bench
    tests/bench_kernels.cpp
  )

  target_link_libraries(stabrank_bench PRIVATE
    stabrank_core_lib
    Catch2::Catch2WithMain
  )
endif()
