project(TinyXML2) 

enable_language(CXX)
	
file(GLOB ${PROJECT_NAME}_sources
	tinyxml2.cpp
	tinyxml2.h
	)

include_directories(${CMAKE_SOURCE_DIR})

add_library(${PROJECT_NAME} STATIC
				${${PROJECT_NAME}_sources}
				)