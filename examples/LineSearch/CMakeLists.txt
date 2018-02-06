cmake_minimum_required(VERSION 3.5)

project(LineSearch)

add_executable(Bracketing ${PROJECT_SOURCE_DIR}/Bracketing/Bracketing.cpp)
add_executable(Brents ${PROJECT_SOURCE_DIR}/Brents/Brents.cpp)

#iterateChildren(${PROJECT_SOURCE_DIR})