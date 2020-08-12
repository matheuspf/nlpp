option(${lib_name_upper}_COVERAGE "If coverage tests will be executed" ON)


add_executable(${lib_name}_tests "")

target_link_libraries(${lib_name}_tests PUBLIC ${lib_name})
target_compile_options(${lib_name}_tests PRIVATE -O2)

if(${${lib_name_upper}_COVERAGE} AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${lib_name}_tests PRIVATE -g --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(${lib_name}_tests PUBLIC -lgcov)
endif()


set(${lib_name}_test_files
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/finite_difference.cpp
)

target_sources(${lib_name}_tests PUBLIC ${${lib_name}_test_files})

enable_testing()
find_package(GTest QUIET)

if(GTest_FOUND)
   target_link_libraries(${lib_name}_tests PUBLIC GTest::GTest GTest::Main)
else()
    message(STATUS "GTest not found. Downloading and installing now.")
    include(${CMAKE_CURRENT_SOURCE_DIR}/external/gtest.cmake EXCLUDE_FROM_ALL)
endif()

add_test(allTests ${lib_name}_tests)