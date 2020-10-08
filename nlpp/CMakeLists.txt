include(get_project_version)

### Add header only library
set(detail_header_files
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/forward_declarations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/helpers_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/include.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/traits.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/types.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/constraints.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/constraints_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/domain.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/domain_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/functions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/functions_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/wrappers/helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/finite_difference_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/optimizer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/output.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/spectraWrapper.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/stop.hpp
)
set(header_files
    ${CMAKE_CURRENT_SOURCE_DIR}/cg/cg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cg/cg_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cg/projections.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cg/projections_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/newton/factorizations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/newton/factorizations_dec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/newton/newton.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/newton/newton_dec.hpp
)
set(header_folders
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/cg
    ${CMAKE_CURRENT_SOURCE_DIR}/newton
)

# Export variables to parent scope
set(detail_header_files ${detail_header_files} PARENT_SCOPE)
set(header_files ${header_files} PARENT_SCOPE)
set(header_folders ${header_folders} PARENT_SCOPE)


add_library(nlpp INTERFACE)
add_library(nlpp::nlpp ALIAS nlpp)

target_compile_features(nlpp INTERFACE cxx_std_17)

target_sources(nlpp INTERFACE "$<BUILD_INTERFACE:${detail_header_files};${header_files}>")

target_include_directories(nlpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${NLPP_INCLUDE_INSTALL_DIR}>
)

target_link_libraries(nlpp
    INTERFACE Eigen3::Eigen
    INTERFACE Spectra::Spectra
    INTERFACE handy::handy
)

if(${NLPP_USE_NANOFLANN})
    target_link_libraries(nlpp
        INTERFACE nanoflann::nanoflann
    )
    target_compile_definitions(nlpp
        INTERFACE NLPP_USE_NANOFLANN=1
    )
endif()


install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${NLPP_INCLUDE_INSTALL_DIR}/nlpp
)

getProjectVersion(${CMAKE_CURRENT_SOURCE_DIR}/helpers/config.hpp)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/nlpp-config-version.cmake
    VERSION ${lib_full_version}
    COMPATIBILITY SameMajorVersion
)

install(TARGETS nlpp
    EXPORT nlpp-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT nlpp-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlpp
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nlpp-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nlpp
)