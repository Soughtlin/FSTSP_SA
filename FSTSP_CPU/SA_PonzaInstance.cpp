# include "SA.h"
#include "read_PonzaInstance.h"
#include "metrics.h"
#include "mpi_utils.h"


// —— 读取 .dat 文件 ——
string PonzaInstance = "PonzaInstances";
const vector<string> PonzaInstance_files = {
        "Instance_005.1_new.dat",
        "Instance_005.2_new.dat",
        "Instance_005.3_new.dat",
        "Instance_005.4_new.dat",
        "Instance_005.5_new.dat",
        "Instance_006.1_new.dat",
        "Instance_006.2_new.dat",
        "Instance_006.3_new.dat",
        "Instance_006.4_new.dat",
        "Instance_006.5_new.dat",
        "Instance_007.1_new.dat",
        "Instance_007.2_new.dat",
        "Instance_007.3_new.dat",
        "Instance_007.4_new.dat",
        "Instance_007.5_new.dat",
        "Instance_008.1_new.dat",
        "Instance_008.2_new.dat",
        "Instance_008.3_new.dat",
        "Instance_008.4_new.dat",
        "Instance_008.5_new.dat",
        "Instance_009.1_new.dat",
        "Instance_009.2_new.dat",
        "Instance_009.3_new.dat",
        "Instance_009.4_new.dat",
        "Instance_009.5_new.dat",
        "Instance_010.1_new.dat",
        "Instance_010.2_new.dat",
        "Instance_010.3_new.dat",
        "Instance_010.4_new.dat",
        "Instance_010.5_new.dat",
        "Instance_050.1_new.dat",
        "Instance_050.2_new.dat",
        "Instance_050.3_new.dat",
        "Instance_050.4_new.dat",
        "Instance_050.5_new.dat",
        "Instance_100.1_new.dat",
        "Instance_100.2_new.dat",
        "Instance_100.3_new.dat",
        "Instance_100.4_new.dat",
        "Instance_100.5_new.dat",
        "Instance_150.1_new.dat",
        "Instance_150.2_new.dat",
        "Instance_150.3_new.dat",
        "Instance_150.4_new.dat",
        "Instance_150.5_new.dat",
        "Instance_200.1_new.dat",
        "Instance_200.2_new.dat",
        "Instance_200.3_new.dat",
        "Instance_200.4_new.dat",
        "Instance_200.5_new.dat"
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

    string instance = PonzaInstance;
    vector<string> files = PonzaInstance_files;
    const int R = 10;

    vector<Metrics> metrics;
    ofstream resultFile("result_PonzaInstance.txt",ios::out | ios::app);


    // 给每列预留固定宽度，并左对齐
    resultFile << left
               << setw(10) << "Instance"
               << setw(15) << "Obj"
               << setw(15) << "CPU Time(ms)"
               << "\n";

    int i = 0;
    for (auto &file: files) {
        InstStat st;
        if (rank == 0) {
            st.inst = readPonzaInstance(instance + "/" + file);
            cerr << "[info] solving " << st.inst.name << " (c=" << st.inst.c << ")" << " Instance " << ++i << "\n";
        }
        broadcastInstance(st, rank, 0);

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
    }

    resultFile.close();

    MPI_Finalize();
    return 0;
}