# project(nlpp::lib VERSION ${lib_name_version} LANGUAGES CXX)

set(lib_headers include/cg/cg.hpp include/cg/projections.hpp)
set(lib_sources src/cg/cg.cpp src/cg/projections.cpp)

add_library(nlpp::lib SHARED ${lib_sources})

target_include_directories(nlpp::lib
    PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(nlpp::lib PUBLIC cxx_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

include(GNUInstallDirs)
install(TARGETS JSONUtils
    EXPORT jsonutils-export
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)