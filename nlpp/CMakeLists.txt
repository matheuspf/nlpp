if(NOT ${lib_name_upper}_INCLUDE_INSTALL_DIR)
    if(INCLUDE_INSTALL_DIR)
        set(${lib_name_upper}_INCLUDE_INSTALL_DIR ${INCLUDE_INSTALL_DIR})
    else()
        set(${lib_name_upper}_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
    endif()
endif()

### Add header only library
set(detail_header_files
    ${lib_path}/helpers/forward_declarations.hpp
    ${lib_path}/helpers/helpers.hpp
    ${lib_path}/helpers/helpers_dec.hpp
    ${lib_path}/helpers/include.hpp
    ${lib_path}/helpers/traits.hpp
    ${lib_path}/helpers/types.hpp
    ${lib_path}/utils/wrappers/constraints.hpp
    ${lib_path}/utils/wrappers/constraints_dec.hpp
    ${lib_path}/utils/wrappers/domain.hpp
    ${lib_path}/utils/wrappers/domain_dec.hpp
    ${lib_path}/utils/wrappers/functions.hpp
    ${lib_path}/utils/wrappers/functions_dec.hpp
    ${lib_path}/utils/wrappers/helpers.hpp
    ${lib_path}/utils/finite_difference_dec.hpp
    ${lib_path}/utils/optimizer.hpp
    ${lib_path}/utils/output.hpp
    ${lib_path}/utils/spectraWrapper.hpp
    ${lib_path}/utils/stop.hpp
)
set(header_files
    ${lib_path}/cg/cg.hpp
    ${lib_path}/cg/cg_dec.hpp
    ${lib_path}/cg/projections.hpp
    ${lib_path}/cg/projections_dec.hpp
    ${lib_path/newton/factorizations.hpp}
    ${lib_path/newton/factorizations_dec.hpp}
    ${lib_path/newton/newton.hpp}
    ${lib_path/newton/newton_dec.hpp}
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
    INTERFACE Eigen
    INTERFACE Spectra
)

if(${lib_name_upper}_USE_NANOFLANN)
    target_link_libraries(${lib_name} INTERFACE Nanoflann)
endif()

install(
    DIRECTORY ${header_folders}
    DESTINATION ${${lib_name_upper}_INCLUDE_INSTALL_DIR}/${lib_name}
)

configure_file(
    ${lib_name}/helpers/config.hpp.in
    ${${lib_name_upper}_INCLUDE_INSTALL_DIR}/${lib_name}/helpers/config.hpp
)

return()



foreach(folder ${include_folders})
    install(
        DIRECTORY ${lib_path}/${folder}
        DESTINATION ${NLPP_INCLUDE_INSTALL_DIR}/${lib_name}
    )
endforeach()


install(FILES Helpers/FiniteDifference.h Helpers/ForwardDeclarations.h helpers/helpers.hpp Helpers/Include.h
              Helpers/Optimizer.h Helpers/Output.h Helpers/Parameters.h Helpers/SpectraHelpers.h Helpers/Stop.h
              Helpers/Types.h Helpers/Wrappers.h
        DESTINATION ${NLPP_INCLUDE_INSTALL_DIR}/${lib_name}/Helpers
)

# configure_file(
#     ${PROJECT_SOURCE_DIR}/include/${lib_name}/Helpers/Config.h.in
#     ${NLPP_INCLUDE_INSTALL_DIR}/${lib_name}/Helpers/Config.h
# )


execute_process(COMMAND git submodule update --init --depth 1 -- ${PROJECT_SOURCE_DIR}/include/nlpp/external/handy
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


set(HANDY_INCLUDE_INSTALL_DIR ${NLPP_INCLUDE_INSTALL_DIR}/${lib_name})

add_subdirectory(external/handy)