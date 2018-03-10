project(LightFinder_GPU_DLL) 

enable_language(CXX CUDA)

file(GLOB ${PROJECT_NAME}_sources
	"headers/*.h" 
	"sources/*.cpp" 
	"headers/*.cuh" 
	"sources/*.cu" 
	)

include_directories(${CMAKE_SOURCE_DIR}/dll/headers)
	
add_library(${PROJECT_NAME} STATIC
				${${PROJECT_NAME}_sources}
				)
			
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

set_target_properties(${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
					   
set_target_properties(${PROJECT_NAME} 
					   PROPERTIES POSITION_INDEPENDENT_CODE ON)