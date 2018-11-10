include(${PROJECT_SOURCE_DIR}/examples/cmake/AddExample.cmake)

set(CG_files CG.cpp)

addExample(${CMAKE_CURRENT_SOURCE_DIR} ${CG_files})