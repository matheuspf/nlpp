find_package(pybind11 REQUIRED)

set(PYTHON_DIR ${PROJECT_SOURCE_DIR}/bindings/python)

add_library(nlpy MODULE)

target_sources(nlpy PUBLIC ${PYTHON_DIR}/src/nlpy.cpp)
target_sources(nlpy PUBLIC ${PYTHON_DIR}/src/GradientDescent/GradientDescent.cpp)
target_sources(nlpy PUBLIC ${PYTHON_DIR}/src/CG/CG.cpp)
target_sources(nlpy PUBLIC ${PYTHON_DIR}/src/Helpers/LineSearch.cpp)
target_sources(nlpy PUBLIC ${PYTHON_DIR}/src/Helpers/Stop.cpp)


target_compile_options(nlpy PRIVATE -flto)

target_link_libraries(nlpy PRIVATE pybind11::module)

target_include_directories(nlpy PUBLIC ${PROJECT_SOURCE_DIR}/include/nlpp ${PROJECT_SOURCE_DIR}/include/nlpp/external/Eigen/)

set_target_properties(nlpy PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                      SUFFIX "${PYTHON_MODULE_EXTENSION}")

