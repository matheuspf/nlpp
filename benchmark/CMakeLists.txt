add_executable(bench "")

target_include_directories(bench PUBLIC ${PROJECT_SOURCE_DIR}/include/nlpp)

target_compile_options(bench PRIVATE -std=c++17 -O2)

add_subdirectory(Helpers)
add_subdirectory(TrustRegion)

find_package(benchmark QUIET)


if(benchmark_FOUND)

   target_link_libraries(bench PUBLIC benchmark::benchmark benchmark::benchmark_main)

else()

    message(STATUS "Google Benchmark not found. Downloading and installing now.")

    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Google Benchmark tests")

    include(external/gbench.cmake EXCLUDE_FROM_ALL)

endif()