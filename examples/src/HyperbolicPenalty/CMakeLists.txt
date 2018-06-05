include(${PROJECT_SOURCE_DIR}/examples/cmake/AddExample.cmake)

set(HyberbolicPenalty_FILES CG.cpp)

addExample(${CMAKE_CURRENT_SOURCE_DIR} ${HyberbolicPenalty_FILES})