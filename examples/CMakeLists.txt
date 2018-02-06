cmake_minimum_required(VERSION 3.5)

set(EXAMPLES OPTIMIZATION_EXAMPLES)

project(${EXAMPLES})


include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    message(FATAL_ERROR "C++17 is needed")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")



get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

set(INCLUDE_DIR ${PARENT_DIR}/include)

include_directories(${INCLUDE_DIR})


include(${INCLUDE_DIR}/Helpers.cmake)

iterateChildren(${PROJECT_SOURCE_DIR})