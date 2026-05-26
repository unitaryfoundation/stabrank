nanobind_add_module(stabrank_core
  ${CMAKE_SOURCE_DIR}/python/bindings.cpp
)

target_link_libraries(stabrank_core PRIVATE stabrank_core_lib)

install(TARGETS stabrank_core LIBRARY DESTINATION stabrank)
