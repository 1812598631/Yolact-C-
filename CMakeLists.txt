cmake_minimum_required(VERSION 2.8)
project(orb_yolact)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

#message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
#OPENCV
#find_package(OpenCV 3.4.10 QUIET)
find_package(OpenCV 4.5.1 QUIET)
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(Boost REQUIRED COMPONENTS thread)

find_package(PythonLibs 3.7 REQUIRED)
message(STATUS "Python Lib Found! Are: " ${PYTHON_LIBRARIES})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#设置可执行文件的存放目录
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/include)
#set(PYTHON_INCLUDE_DIRS "/home/run/anaconda3/envs/yolact-env/include/python3.7m/")
set(PYTHON_LIBRARIES "/home/run/anaconda3/envs/yolact-env/lib/libpython3.7m.so")

INCLUDE_DIRECTORIES(
        ${PYTHON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include
#        /home/run/anaconda3/envs/yolact-env/include
#        /home/run/anaconda3/envs/yolact-env/include/python3.7m
)
include_directories(BEFORE "/home/run/anaconda3/envs/yolact-env/lib/python3.7/site-packages/numpy/core/include/numpy")
include_directories(BEFORE "/home/run/anaconda3/envs/yolact-env/lib/python3.7/site-packages/numpy/core/include")
link_directories(/home/run/anaconda3/envs/yolact-env/lib/python3.7/site-packages/numpy/core/lib)
add_library(yolact_interface src/yolact.cc)
target_link_libraries(yolact_interface ${PYTHON_LIBRARIES} ${OpenCV_LIBS} dl util)
add_executable(orb_yolact src/c++test.cpp)
target_link_libraries(orb_yolact yolact_interface)
