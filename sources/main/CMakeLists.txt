project(LightFinder_GPU_EXE) 

enable_language(CXX)
	
file(GLOB ${PROJECT_NAME}_sources
	main.cpp
	main.h
	)

include_directories(${CMAKE_SOURCE_DIR}/sources/dll/headers)

add_executable(${PROJECT_NAME}
				${${PROJECT_NAME}_sources}
				)
				
target_link_libraries(${PROJECT_NAME} LightFinder_GPU_DLL)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/resources/In.xml $<TARGET_FILE_DIR:${PROJECT_NAME}>/In.xml)