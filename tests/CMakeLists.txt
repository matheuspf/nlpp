option(NLPP_COVERAGE "If coverage tests will be executed" ON)


add_executable(nlpp_tests "")

target_link_libraries(nlpp_tests PUBLIC nlpp)
# target_compile_options(nlpp_tests PRIVATE -O2)

if(${NLPP_COVERAGE} AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(nlpp_tests PRIVATE -g --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(nlpp_tests PUBLIC -lgcov)
endif()


set(nlpp_test_files
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/finite_difference.cpp
)

target_sources(nlpp_tests PUBLIC ${nlpp_test_files})

enable_testing()
find_package(GTest QUIET)

if(GTest_FOUND)
   target_link_libraries(nlpp_tests PUBLIC GTest::GTest GTest::Main)
else()
    message(STATUS "GTest not found. Downloading and installing now.")
    include(${CMAKE_CURRENT_SOURCE_DIR}/external/gtest.cmake EXCLUDE_FROM_ALL)
endif()

add_test(allTests nlpp_tests)