project(CUDATests)

enable_language(CXX)

file(GLOB ${PROJECT_NAME}_sources
	"*.cpp" 
	)
	
add_executable(${PROJECT_NAME}
	${${PROJECT_NAME}_sources}
)

target_link_libraries(${PROJECT_NAME} LightFinder_GPU_DLL)
target_link_libraries(${PROJECT_NAME} gtest gtest_main)
target_link_libraries(${PROJECT_NAME} gmock gmock_main)

include_directories(
	${CMAKE_SOURCE_DIR}/dll/headers
	${CMAKE_BINARY_DIR}/googletest-src/googletest/include
	${CMAKE_BINARY_DIR}/googletest-src/googlemock/include
	)