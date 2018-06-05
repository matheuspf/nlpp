cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME HandyTest)

project (${PROJECT_NAME})


find_package(Threads REQUIRED)

find_package(GTest)


if(NOT ${GTEST_FOUND})
  add_subdirectory(gtest)
endif()

enable_testing()


include(CheckCXXCompilerFlag)


CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)

if(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

else()
  message( FATAL_ERROR "Handy cannot be compiled on ${CMAKE_CXX_COMPILER} because it does not support C++11")

endif()


set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O2 -g -fprofile-arcs -ftest-coverage -Wno-deprecated -Wno-unused-value -pthread")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

include_directories(${GTEST_INCLUDE_DIRS})


FILE(GLOB children RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/*)

foreach(child ${children})
  if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${child} and
     exists ${PROJECT_SOURCE_DIR}/${child}/CMakeLists.txt and
     not ${child} strequal "build" and
     not ${child} strequal "gtest")

     if(  )
     include(${child}/CMakeLists.txt)

    endif()
endforeach()



add_executable(${PROJECT_NAME} ${TEST_SRC_FILES})


if(NOT ${GTEST_FOUND})

  add_dependencies(${PROJECT_NAME} googletest)
  
  set(GTEST_LIBRARIES ${GTEST_LIBS_DIR}/libgtest.a)
  set(GTEST_MAIN_LIBRARIES ${GTEST_LIBS_DIR}/libgtest_main.a)

endif()


target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})


add_test(test ${PROJECT_NAME})