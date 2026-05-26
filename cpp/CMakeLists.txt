include(FetchContent)

# --- Eigen (header-only linear algebra) ---
find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT Eigen3_FOUND)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
  )
  # Eigen's CMakeLists builds tests/docs by default; disable them without
  # permanently disabling this project's CTest registration.
  set(STABRANK_BUILD_TESTING "${BUILD_TESTING}")
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
  set(BUILD_TESTING "${STABRANK_BUILD_TESTING}" CACHE BOOL "" FORCE)
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
)

target_include_directories(stabrank_core_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(stabrank_core_lib PUBLIC Eigen3::Eigen)

if(BUILD_TESTING)
  # --- Catch2 tests ---
  add_executable(stabrank_tests
    tests/test_linalg.cpp
    tests/test_polynomial.cpp
    tests/test_pauli.cpp
    tests/test_sa_engine.cpp
    tests/test_kernel_properties.cpp
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
