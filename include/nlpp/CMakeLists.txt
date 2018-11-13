set(include_folders CG GradientDescent HyperbolicPenalty LineSearch Newton QuasiNewton TestFunctions TrustRegion)

if(NOT NLPP_INCLUDE_INSTALL_DIR)
    if(INCLUDE_INSTALL_DIR)
        set(NLPP_INCLUDE_INSTALL_DIR ${INCLUDE_INSTALL_DIR})
    else()
        set(NLPP_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include)
    endif()
endif()


foreach(folder ${include_folders})
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/${lib_name}/${folder}
        DESTINATION ${NLPP_INCLUDE_INSTALL_DIR}/${lib_name}
    )
endforeach()

install(FILES Helpers/FiniteDifference.h Helpers/ForwardDeclarations.h Helpers/Helpers.h Helpers/Include.h
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