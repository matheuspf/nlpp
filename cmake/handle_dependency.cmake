function(handleDependency dep_name submodule_name)
    find_package(${dep_name} QUIET)
    if(${dep_name}_FOUND)
        # set(NLPP_HAS_IMPORTED_TARGETS ON)
        message(${dep_name} " Found")
    else()
        message(${dep_name} " NOT Found")
        # set(NLPP_HAS_IMPORTED_TARGETS OFF)
        if(TARGET ${dep_name})
            message(STATUS "Using inherited ${dep} target")
        else()
            message(STATUS "Installing ${dep} via submodule")

            execute_process(COMMAND git submodule update --init --depth 1 -- external/${submodule_name}
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nlpp)
            execute_process(COMMAND git checkout .
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nlpp/external/${submodule_name})

            set(patch_path ${CMAKE_CURRENT_SOURCE_DIR}/nlpp/external/patches/${submodule_name}.patch)
            if(EXISTS ${patch_path})
                execute_process(COMMAND git apply ${patch_path}
                                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nlpp/external/${submodule_name})
            endif()
            add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nlpp/external/${submodule_name} EXCLUDE_FROM_ALL)
        endif()
    endif()
endfunction()