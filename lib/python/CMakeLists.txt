find_package(pybind11 REQUIRED)

add_library(nlpy MODULE src/test.cpp)

target_compile_options(nlpy PRIVATE -flto)

target_link_libraries(nlpy PRIVATE pybind11::module)

target_include_directories(nlpy PUBLIC ${PROJECT_SOURCE_DIR}/include/nlpp ${PROJECT_SOURCE_DIR}/include/nlpp/external/Eigen/)

set_target_properties(nlpy PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                      SUFFIX "${PYTHON_MODULE_EXTENSION}")

