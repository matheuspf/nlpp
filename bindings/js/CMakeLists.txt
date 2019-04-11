cmake_minimum_required(VERSION 3.5)

set(base_path ${SOURCE_DIR}../../)

set(lib_name js_nlp)

project(${lib_name} VERSION 1.0.0 LANGUAGES CXX)


# find_package(Threads


### Add library
add_executable(${lib_name} src/Optimizer.cpp)

target_include_directories(${lib_name} PUBLIC ${base_path}/include ${base_path}/include/nlpp include)

target_compile_options(${lib_name} PUBLIC -std=c++14 -O2)

target_link_libraries(${lib_name} PUBLIC --bind)