cmake_minimum_required(VERSION 3.25)
project(FSTSP_serial)

set(CMAKE_CXX_STANDARD 17)

add_executable(SA_MurrayInstance SA_MurrayInstance.cpp instance.cpp metrics.cpp read_MurryInstance.cpp SA.cpp)
add_executable(SA_PonzaInstance SA_PonzaInstance.cpp instance.cpp metrics.cpp read_PonzaInstance.cpp SA.cpp)
