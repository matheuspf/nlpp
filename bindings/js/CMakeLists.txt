cmake_minimum_required(VERSION 3.5)

set(lib_name js_nlp)

string(TOUPPER ${lib_name} lib_name_upper)

### So we can install without building everything (docs, examples and tests)
#set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)


### Create the project with the given version
project(${lib_name} VERSION 1.0.0 LANGUAGES CXX)



### Add library
add_executable(${lib_name} src/Optimizer.cpp)

target_include_directories(${lib_name} PUBLIC ../include include)

#target_sources(${lib_name} PRIVATE t1.cpp)

# target_compile_options(${lib_name} PUBLIC -std=c++14 -O3 -s WASM=1 -DNDEBUG)
target_compile_options(${lib_name} PUBLIC -std=c++14 -O2 -Wno-delete-non-virtual-dtor)

target_link_libraries(${lib_name} PUBLIC --bind)