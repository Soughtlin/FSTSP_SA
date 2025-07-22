cmake_minimum_required(VERSION 3.25)
project(src_para_cpu_gpu)

set(CMAKE_CXX_STANDARD 17)

add_executable(SA_MurrayInstance.cpp instance.cpp metrics.cpp read_MurrayInstance.cpp SA.cpp)
add_executable(SA_PonzaInstance.cpp instance.cpp metrics.cpp read_PonzaInstance.cpp SA.cpp)
