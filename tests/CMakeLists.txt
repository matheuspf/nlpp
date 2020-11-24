Include(FetchContent)

option(NLPP_COVERAGE "If coverage tests will be executed" ON)


add_executable(nlpp_tests "")

target_link_libraries(nlpp_tests PUBLIC nlpp)
# target_compile_options(nlpp_tests PRIVATE -O2)

if(${NLPP_COVERAGE} AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(nlpp_tests PRIVATE -g --coverage -fprofile-arcs -ftest-coverage)
    target_link_libraries(nlpp_tests PUBLIC -lgcov)
endif()


set(nlpp_test_files
    # ${CMAKE_CURRENT_SOURCE_DIR}/utils/finite_difference.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/utils/output.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/utils/stop.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/constraints.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/domain.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/optimizer.cpp
)

target_sources(nlpp_tests PUBLIC ${nlpp_test_files})

enable_testing()
# find_package(GTest QUIET)

# if(GTest_FOUND)
#    target_link_libraries(nlpp_tests PUBLIC GTest::GTest GTest::Main)
# else()
#     message(STATUS "GTest not found. Downloading and installing now.")
#     include(${CMAKE_CURRENT_SOURCE_DIR}/external/gtest.cmake EXCLUDE_FROM_ALL)
# endif()

# add_test(allTests nlpp_tests)


find_package(Catch2 QUIET)

if(NOT Catch2_FOUND)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        devel
    )
    FetchContent_MakeAvailable(Catch2)
endif()

target_link_libraries(nlpp_tests PUBLIC Catch2::Catch2WithMain)

add_test(allTests nlpp_tests)


find_package(boost_mp11 QUIET)

if(NOT boost_mp11_FOUND)
    FetchContent_Declare(
        boost_mp11
        GIT_REPOSITORY https://github.com/boostorg/mp11
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(boost_mp11)
endif()


target_link_libraries(nlpp_tests PUBLIC Boost::mp11)
