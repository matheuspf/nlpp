function(addExample file_path files)

    foreach(file_name ${files})

        get_filename_component(file ${file_name} NAME_WE)

        add_executable(${file} ${file_path}/${file_name})

        target_include_directories(${file} PUBLIC ${PROJECT_SOURCE_DIR}/include/nlpp)

        target_compile_features(${file} PRIVATE cxx_std_17)

        target_compile_options(${file} PRIVATE -O2)

        target_compile_definitions(${file} PUBLIC EIGEN_INCLUDE_LOCAL=${_eigen_include_local})

        add_dependencies(examples ${file})
    endforeach()

endfunction()