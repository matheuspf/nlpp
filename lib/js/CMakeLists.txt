cmake_minimum_required(VERSION 3.5)

set(lib_name js_nlp)

string(TOUPPER nlpp lib_name_upper)

### So we can install without building everything (docs, examples and tests)
#set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)


### Create the project with the given version
project(nlpp VERSION 1.0.0 LANGUAGES CXX)



### Add library
add_executable(nlpp src/Optimizer.cpp)

target_include_directories(nlpp PUBLIC ../include include)

#target_sources(nlpp PRIVATE t1.cpp)

# target_compile_options(nlpp PUBLIC -std=c++14 -O3 -s WASM=1 -DNDEBUG)
target_compile_options(nlpp PUBLIC -std=c++14 -O2 -Wno-delete-non-virtual-dtor)

target_link_libraries(nlpp PUBLIC --bind)