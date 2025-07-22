# include "SA.h"
#include "read_MurryInstance.h"
#include "metrics.h"
#include "mpi_utils.h"
#include "cuda_evaluate.cuh"


// Murray Instance
string MurrayInstance = "MurrayInstance";
vector<string> MurrayInstance_folders = {
        "20140810T123437v1",
        "20140810T123437v2",
        "20140810T123437v3",
        "20140810T123437v4",
        "20140810T123437v5",
        "20140810T123437v6",
        "20140810T123437v7",
        "20140810T123437v8",
        "20140810T123437v9",
        "20140810T123437v10",
        "20140810T123437v11",
        "20140810T123437v12",
        "20140810T123440v1",
        "20140810T123440v2",
        "20140810T123440v3",
        "20140810T123440v4",
        "20140810T123440v5",
        "20140810T123440v6",
        "20140810T123440v7",
        "20140810T123440v8",
        "20140810T123440v9",
        "20140810T123440v10",
        "20140810T123440v11",
        "20140810T123440v12",
        "20140810T123443v1",
        "20140810T123443v2",
        "20140810T123443v3",
        "20140810T123443v4",
        "20140810T123443v5",
        "20140810T123443v6",
        "20140810T123443v7",
        "20140810T123443v8",
        "20140810T123443v9",
        "20140810T123443v10",
        "20140810T123443v11",
        "20140810T123443v12",
};


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long long global_seed;
    if (rank == 0) {
        std::random_device rd;
        global_seed = (static_cast<unsigned long long>(rd()) << 32) ^ rd();
    }
    MPI_Bcast(&global_seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD); //把种子广播给所有进程

    string instance = MurrayInstance;
    vector<string> folders = MurrayInstance_folders;
    const int R = 10;

    vector<Metrics> metrics;
#ifdef USE_CUDA
    ofstream resultFile("result_MurrayInstance_usecuda.txt",ios::out | ios::app);
#else
    ofstream resultFile("result_MurrayInstance.txt",ios::out | ios::app);
#endif

    // 给每列预留固定宽度，并左对齐
    resultFile << left
               << setw(10) << "Instance"
               << setw(15) << "Obj"
               << setw(15) << "CPU Time(ms)"
               << "\n";

    int i = 0;
    for (auto &folder: folders) {
        InstStat st;
        if (rank == 0) {
            st.inst = readMurrayInstance(instance + "/" + folder);
            cerr << "[info] solving " << st.inst.name << " (c=" << st.inst.c << ")" << " Instance " << ++i << "\n";
        }
        broadcastInstance(st, rank, 0);

#ifdef USE_CUDA
        // —— 在这里初始化 GPU 端的 tauTruck/tauDrone/SL/SR 常量 ——
        init_gpu_constants(st.inst);
#endif

        // === 多进程并行 R 条链 ===
        int base = R / size, extra = R % size;
        int myCnt = base + (rank < extra);
        int start = rank * base + std::min(rank, extra);

        std::vector<RunStat> localRuns(myCnt);
        for (int k = 0; k < myCnt; ++k) {
            int chainId = start + k;
            seed_rng(global_seed + chainId);
            localRuns[k] = solveInstance(st.inst);
        }

        // === Gather RunStat 到 root ===
        MPI_Barrier(MPI_COMM_WORLD);
        gatherRunStats(localRuns, R, rank, size, st.runs);

        // ---- 计算指标 ----
        if (rank == 0) {
            Metrics M = computeMetrics(st, R);
            metrics.push_back(M);
            cout << fixed << setprecision(4);
            cout << "obj best = " << M.obj_best << "\n";
            cout << "CPU time Avg = " << M.cpuAvg << " ms" << endl;
        }
    }

    // 最后统一写文件，每列宽度和表头保持一致
    resultFile << fixed << setprecision(4);
    for (int j = 0; j < metrics.size(); ++j) {
        resultFile << left
                   << setw(10) << (j + 1)
                   << setw(15) << metrics[j].obj_best
                   << setw(15) << metrics[j].cpuAvg
                   << "\n";

#ifdef USE_CUDA
        // —— 在每个实例跑完之后，释放 GPU 常量区内存 ——
        free_gpu_constants();
#endif

    }

    resultFile.close();

    MPI_Finalize();
    return 0;
}