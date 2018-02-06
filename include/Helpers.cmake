function(iterateChildren CUR_PATH)

    file(GLOB CHILDREN RELATIVE ${CUR_PATH} ${CUR_PATH}/*)

    foreach(CHILD ${CHILDREN})
        set(CHILD_DIR ${PROJECT_SOURCE_DIR}/${CHILD})

        if(IS_DIRECTORY ${CHILD_DIR} AND NOT ${CHILD} STREQUAL "build")

            if(EXISTS ${CHILD_DIR}/CMakeLists.txt)
                add_subdirectory(${CHILD_DIR})
            else()
                file(GLOB CHILD_SRC ${CHILD_DIR}/*.cpp)
                add_executable(${CHILD} ${CHILD_SRC})
            endif()
            
        endif()
    endforeach()

endfunction()