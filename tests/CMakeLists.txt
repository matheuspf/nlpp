option(${lib_name_upper}_COVERAGE "If coverage tests will be executed" ON)


add_executable(tests "")

target_include_directories(tests PUBLIC ${PROJECT_SOURCE_DIR}/include/nlpp)

target_compile_options(tests PRIVATE -std=c++17 -O2)


if(${lib_name_upper}_COVERAGE AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(tests PRIVATE -g --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(tests PUBLIC -lgcov)
endif()


#add_subdirectory(CG)
#add_subdirectory(GradientDescent)
#add_subdirectory(LineSearch)
#add_subdirectory(QuasiNewton)
#add_subdirectory(HyperbolicPenalty)
#add_subdirectory(Newton)
add_subdirectory(GradientOptimizer)


enable_testing()

find_package(GTest QUIET)


if(GTest_FOUND)

   target_link_libraries(tests PUBLIC GTest::GTest GTest::Main)

else()

    message(STATUS "GTest not found. Downloading and installing now.")

    include(external/gtest.cmake EXCLUDE_FROM_ALL)

endif()

add_test(allTests tests)