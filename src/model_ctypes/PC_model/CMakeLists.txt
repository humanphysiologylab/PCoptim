cmake_minimum_required(VERSION 3.8)	 
project(PC_model)


include(FindPkgConfig)
pkg_search_module(CVODE REQUIRED CVODE>=6.0.0)

include_directories(${CVODE_INCLUDE_DIRS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm -lsundials_cvode")


set(SOURCE_FILES main.c ina.c)
add_executable(main ${SOURCE_FILES})

target_link_libraries(main ina ${CVODE_LIBRARIES})
