cmake_minimum_required(VERSION 3.13)
project(NN0_1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
set(SOURCE_FILES CppFiles/main.cpp)


add_executable(NN0_1 CppFiles/main.cpp CppFiles/FeedForwardNetwork.h CppFiles/helperFunctions.h CppFiles/FeedForwardNetwork.cpp CppFiles/helperFunctions.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(NN0_1 ${OpenCV_LIBS})
