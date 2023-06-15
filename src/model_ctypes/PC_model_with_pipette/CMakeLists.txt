cmake_minimum_required(VERSION 3.8)  
project(PC_model)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(SUNDIALS 6.0.0 REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm -lsundials_cvode")

set(SOURCE_FILES ina.c run.c)

add_library(ina SHARED ${SOURCE_FILES})
target_link_libraries(ina m sundials_cvode)
