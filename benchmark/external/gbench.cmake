
execute_process(COMMAND git submodule update --init -- ${PROJECT_SOURCE_DIR}/benchmark/external/googlebench
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/benchmark/external/googlebench)

target_link_libraries(bench PRIVATE benchmark benchmark_main)