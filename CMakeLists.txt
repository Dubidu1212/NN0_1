cmake_minimum_required(VERSION 3.13)

project(NN0_1)


set(GCC_COVERAGE_COMPILE_FLAGS "-D WITH_TBB=ON -D WITH_OPENMP=ON -D WITH_IPP=ON -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_EXAMPLES=OFF -D WITH_NVCUVID=ON -D WITH_CUDA=ON -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_CSTRIPES=ON -D WITH_OPENCL=ON ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)



add_executable(NN0_1 CppFiles/helperFunctions.h CppFiles/helperFunctions.cpp  CppFiles/FeedForwardNetwork.h CppFiles/FeedForwardNetwork.cpp CppFiles/main.cpp CppFiles/ConvolutionalNetwork.cpp CppFiles/ConvolutionalNetwork.h CppFiles/NetworkLayer.h CppFiles/ConvolutionalLayer.cpp CppFiles/ConvolutionalLayer.h CppFiles/ReLULayer.cpp CppFiles/ReLULayer.h CppFiles/MaxPoolLayer.cpp CppFiles/MaxPoolLayer.h CppFiles/FullyConnectedLayer.cpp CppFiles/FullyConnectedLayer.h CppFiles/EvolutionMaster.cpp CppFiles/EvolutionMaster.h CppFiles/gene.cpp CppFiles/gene.h CppFiles/DropoutLayer.cpp CppFiles/DropoutLayer.h)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(NN0_1 ${OpenCV_LIBS})
