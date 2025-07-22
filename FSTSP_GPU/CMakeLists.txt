cmake_minimum_required(VERSION 3.25)
project(src)

set(CMAKE_CXX_STANDARD 17)

add_executable(SA_MurrayInstance instance.cpp metrics.cpp read_MurrayInstance.cpp SA.cpp)
add_executable(SA_PonzaInstance SA_PonzaInstance.cu instance.cpp metrics.cpp read_PonzaInstance.cpp SA.cpp)
