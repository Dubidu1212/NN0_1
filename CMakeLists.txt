cmake_minimum_required(VERSION 3.13)

project(NN0_1)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)



add_executable(NN0_1 CppFiles/helperFunctions.h CppFiles/helperFunctions.cpp  CppFiles/FeedForwardNetwork.h CppFiles/FeedForwardNetwork.cpp CppFiles/main.cpp CppFiles/ConvolutionalNetwork.cpp CppFiles/ConvolutionalNetwork.h CppFiles/layerTemplate.cpp CppFiles/layerTemplate.h CppFiles/NetworkLayer.h CppFiles/ConvolutionalLayer.cpp CppFiles/ConvolutionalLayer.h CppFiles/ReLULayer.cpp CppFiles/ReLULayer.h CppFiles/MaxPoolLayer.cpp CppFiles/MaxPoolLayer.h)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(NN0_1 ${OpenCV_LIBS})
