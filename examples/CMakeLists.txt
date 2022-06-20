function(addExamples file_name_list)
    foreach(file_name ${file_name_list})
        get_filename_component(file ${file_name} NAME_WE)
        add_executable(nlpp_${file}_example ${file_name})
        target_link_libraries(nlpp_${file}_example PUBLIC nlpp)
        target_compile_options(nlpp_${file}_example PRIVATE -O2)
        add_dependencies(nlpp_examples nlpp_${file}_example)
    endforeach()
endfunction()

add_custom_target(nlpp_examples)


set(nlpp_example_files
    # ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/constraints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/domain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/functions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/stop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/finite_difference.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/line_search/backtracking/backtracking.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/line_search/goldstein/goldstein.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/line_search/strong_wolfe/strong_wolfe.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/cg/cg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/newton/newton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quasi_newton/bfgs/bfgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quasi_newton/lbfgs/lbfgs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyberbolic_penalty/hyberbolic_penalty.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/citgo/knn/knn.cpp
)

addExamples("${nlpp_example_files}")
