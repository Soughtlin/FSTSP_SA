//
// Created by Song.H.L on 2025/7/6.
//

#include "mpi_utils.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda_evaluate.cuh"
#endif

void broadcastInstance(InstStat &st, int rank, int root = 0) {
    // 先广播 c
    int c;
    if (rank == root) c = st.inst.c;
    MPI_Bcast(&c, 1, MPI_INT, root, MPI_COMM_WORLD);

    if (rank != root) {
        st.inst.c = c;
        st.inst.tauTruck.assign(c + 2, std::vector<double>(c + 2));
        st.inst.tauDrone.assign(c + 2, std::vector<double>(c + 2));
    }

    // 3 个标量
    double scalars[3];
    if (rank == root) {
        scalars[0] = st.inst.SL;
        scalars[1] = st.inst.SR;
        scalars[2] = st.inst.droneEndurance;
    }
    MPI_Bcast(scalars, 3, MPI_DOUBLE, root, MPI_COMM_WORLD);
    if (rank != root) {
        st.inst.SL = scalars[0];
        st.inst.SR = scalars[1];
        st.inst.droneEndurance = scalars[2];
    }

    // 两张 (c+2)×(c+2) 矩阵
    const int n = (c + 2) * (c + 2);
    std::vector<double> buf(n);

    // tauTruck
    if (rank == root) {
        for (int i = 0; i < c + 2; ++i) {
            std::copy(
                    st.inst.tauTruck[i].begin(),
                    st.inst.tauTruck[i].end(),
                    buf.begin() + i * (c + 2)
            );
        }
    }
    MPI_Bcast(buf.data(), n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    if (rank != root) {
        for (int i = 0; i < c + 2; ++i) {
            std::copy(
                    buf.begin() + i * (c + 2),
                    buf.begin() + (i + 1) * (c + 2),
                    st.inst.tauTruck[i].begin()
            );
        }
    }

    // tauDrone（复用 buf）
    if (rank == root) {
        for (int i = 0; i < c + 2; ++i) {
            std::copy(
                    st.inst.tauDrone[i].begin(),
                    st.inst.tauDrone[i].end(),
                    buf.begin() + i * (c + 2)
            );
        }
    }
    MPI_Bcast(buf.data(), n, MPI_DOUBLE, root, MPI_COMM_WORLD);
    if (rank != root) {
        for (int i = 0; i < c + 2; ++i) {
            std::copy(
                    buf.begin() + i * (c + 2),
                    buf.begin() + (i + 1) * (c + 2),
                    st.inst.tauDrone[i].begin()
            );
        }
    }

    // BKS
    MPI_Bcast(&st.inst.BKS, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

    // 实例名
    int len;
    if (rank == root) len = static_cast<int>(st.inst.name.size());
    MPI_Bcast(&len, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank != root) st.inst.name.resize(len);
    MPI_Bcast(const_cast<char*>(st.inst.name.data()), len, MPI_CHAR, root, MPI_COMM_WORLD);

    // —— 新增：GPU 常量初始化 ——
#ifdef USE_CUDA
    // 把 tauTruck/tauDrone/SL/SR 一次性拷到 GPU
      init_gpu_constants(st.inst);
#endif
}

void gatherRunStats(const std::vector<RunStat> &localRuns,
                    int totalR, int rank, int size,
                    std::vector<RunStat> &allRuns) {
    // —— 新增：确保所有 GPU 计算完成 ——
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif

    const int sendBytes = static_cast<int>(localRuns.size() * sizeof(RunStat));

    // 先把每个进程要发送的字节数收集到 root
    std::vector<int> recvBytes;
    if (rank == 0) recvBytes.resize(size);

    MPI_Gather(&sendBytes, 1, MPI_INT,
               rank==0 ? recvBytes.data() : nullptr,
               1, MPI_INT, 0, MPI_COMM_WORLD);

    // root 计算位移数组
    std::vector<int> displs;
    if (rank == 0) {
        displs.resize(size);
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = offset;
            offset   += recvBytes[i];
        }
        allRuns.resize(totalR);
    }

    // 按字节收集所有 RunStat
    MPI_Gatherv(localRuns.data(), sendBytes, MPI_BYTE,
                rank==0 ? allRuns.data() : nullptr,
                rank==0 ? recvBytes.data() : nullptr,
                rank==0 ? displs.data()   : nullptr,
                MPI_BYTE, 0, MPI_COMM_WORLD);
}
