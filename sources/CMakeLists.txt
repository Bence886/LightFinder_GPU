cmake_minimum_required(VERSION 3.8 FATAL_ERROR) 
PROJECT(LightFinder_GPU) 
set(VERSION_MAJOR "1") 
set(VERSION_MINOR "0") 
set(VERSION_PATCH "0") 
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}") 

enable_language(CXX CUDA)

add_subdirectory(dll)
add_subdirectory(main)

add_subdirectory(../dependencies/tinyxml2 ${CMAKE_BINARY_DIR}/dependencies/tinyxml2)
add_subdirectory(../tests ${CMAKE_BINARY_DIR}/tests)

configure_file(../tests/CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
	
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build")