# FSTSP_SA
Solving FSTSP by SA algorithms. Using paralell tools to accelerate the algorithm.

**gpu**

nvcc -O0 -std=c++14 SA_device.cu SA_PonzaInstance.cu instance.cpp SA.cpp read_PonzaInstance.cpp metrics.cpp -lcurand -o SA_PonzaInstance

./SA_PonzaInstance

**gpu+cpu**

nvcc -DUSE_CUDA -O0 -std=c++14 -ccbin mpicxx SA_PonzaInstance.cpp instance.cpp metrics.cpp read_PonzaInstance.cpp SA.cpp mpi_utils.cpp cuda_evaluate.cu -lcudart -lcurand -o SA_PonzaInstance  

mpirun -np 2 ./SA_PonzaInstance

**cpu**

mpicxx -std=c++14 SA_PonzaInstance.cpp instance.cpp metrics.cpp read_PonzaInstance.cpp SA.cpp mpi_utils.cpp -fopenmp -o SA_PonzaInstance

mpirun -np 2 ./SA_PonzaInstance
