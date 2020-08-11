if(NOT ${lib_name_upper}_INCLUDE_INSTALL_DIR)
    if(INCLUDE_INSTALL_DIR)
        set(${lib_name_upper}_INCLUDE_INSTALL_DIR ${INCLUDE_INSTALL_DIR})
    else()
        set(${lib_name_upper}_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
    endif()
endif()

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


add_library(${lib_name} INTERFACE)

target_sources(${lib_name}
    INTERFACE $<BUILD_INTERFACE:${detail_header_files};${header_files}>
)

target_include_directories(${lib_name}
    INTERFACE $<BUILD_INTERFACE:${lib_path}>
    SYSTEM INTERFACE $<INSTALL_INTERFACE:${${lib_name_upper}_INCLUDE_INSTALL_DIR}>
)

target_compile_features(${lib_name}
    INTERFACE cxx_std_17
)

target_link_libraries(${lib_name}
    INTERFACE handy
    INTERFACE Eigen3::Eigen
    INTERFACE Spectra::Spectra
)

if(${${lib_name_upper}_USE_NANOFLANN})
    target_link_libraries(${lib_name}
        INTERFACE nanoflann::nanoflann
    )
    target_compile_definitions(${lib_name}
        INTERFACE ${lib_name_upper}_USE_NANOFLANN=1
    )
endif()


install(
    DIRECTORY ${header_folders}
    DESTINATION ${${lib_name_upper}_INCLUDE_INSTALL_DIR}/${lib_name}
)
