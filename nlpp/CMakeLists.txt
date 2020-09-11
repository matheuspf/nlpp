### Add header only library
set(detail_header_files
    helpers/forward_declarations.hpp
    helpers/helpers.hpp
    helpers/helpers_dec.hpp
    helpers/include.hpp
    helpers/traits.hpp
    helpers/types.hpp
    utils/wrappers/constraints.hpp
    utils/wrappers/constraints_dec.hpp
    utils/wrappers/domain.hpp
    utils/wrappers/domain_dec.hpp
    utils/wrappers/functions.hpp
    utils/wrappers/functions_dec.hpp
    utils/wrappers/helpers.hpp
    utils/finite_difference_dec.hpp
    utils/optimizer.hpp
    utils/output.hpp
    utils/spectraWrapper.hpp
    utils/stop.hpp
)
set(header_files
    cg/cg.hpp
    cg/cg_dec.hpp
    cg/projections.hpp
    cg/projections_dec.hpp
    newton/factorizations.hpp
    newton/factorizations_dec.hpp
    newton/newton.hpp
    newton/newton_dec.hpp
)
set(header_folders
    ${lib_path}/helpers
    ${lib_path}/utils
    ${lib_path}/cg
    ${lib_path}/newton
)

# Export variables to parent scope
set(detail_header_files ${detail_header_files} PARENT_SCOPE)
set(header_files ${header_files} PARENT_SCOPE)
set(header_folders ${header_folders} PARENT_SCOPE)


add_library(nlpp INTERFACE)
add_library(nlpp::nlpp ALIAS nlpp)

target_compile_features(nlpp INTERFACE cxx_std_17)

target_sources(nlpp INTERFACE
    $<BUILD_INTERFACE:${detail_header_files};${header_files}>
)

target_include_directories(nlpp INTERFACE
    $<BUILD_INTERFACE:${lib_path}>
    $<INSTALL_INTERFACE:${NLPP_INCLUDE_INSTALL_DIR}>
)

target_link_libraries(nlpp
    INTERFACE handy::handy
    INTERFACE Eigen3::Eigen
    INTERFACE Spectra::Spectra
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
    DIRECTORY ${lib_path}/
    DESTINATION ${NLPP_INCLUDE_INSTALL_DIR}/nlpp
)

getProjectVersion(nlpp ${lib_path}/helpers/config.hpp)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/nlpp-config-version.cmake
    VERSION ${nlpp_full_version}
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