include(${PROJECT_SOURCE_DIR}/examples/cmake/AddExample.cmake)

set(Wrappers_files Wrappers.cpp)

addExample(${CMAKE_CURRENT_SOURCE_DIR} ${Wrappers_files})