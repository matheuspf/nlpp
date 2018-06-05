include(${PROJECT_SOURCE_DIR}/examples/cmake/AddExample.cmake)

set(FiniteDifference_files FiniteDifference.cpp)

addExample(${CMAKE_CURRENT_SOURCE_DIR} ${FiniteDifference_files})