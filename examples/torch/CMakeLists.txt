find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(t1 ${CMAKE_CURRENT_SOURCE_DIR}/t2.cpp)
target_link_libraries(t1 "${TORCH_LIBRARIES}")
target_link_libraries(t1 nlpp)
#set_property(TARGET t1 PROPERTY CXX_STANDARD 17)