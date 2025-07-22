# include "SA.h"
#include "read_PonzaInstance.h"
#include "metrics.h"


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


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string instance = PonzaInstance;
    vector<string> files = PonzaInstance_files;
    const int R = 10;

    vector<Metrics> metrics;
    ofstream resultFile("result_PonzaInstance.txt");

    // 给每列预留固定宽度，并左对齐
    resultFile << left
               << setw(10) << "Instance"
               << setw(15) << "Obj"
               << setw(15) << "CPU Time(ms)"
               << "\n";

    int i = 0;
    for (auto &file: files) {
        InstStat st;
        st.inst = readPonzaInstance(instance + "/" + file);

        // ---- 求解 ----
        cerr << "[info] solving " << st.inst.name << " (c=" << st.inst.c << ")" << " Instance " << ++i << "\n";
        for (int r = 0; r < R; ++r) {
            RunStat rs = solveInstance_test(st.inst);
            st.runs.push_back(rs);
        }

        // ---- 计算指标 ----
        Metrics M = computeMetrics(st, R);
        metrics.push_back(M);  // 把结果写回已分配好的 vector

        // ---- 控制台输出 ----
        cout << fixed << setprecision(4);
        cout << "obj best = " << M.obj_best << "\n";
        cout << "CPU time Avg = " << M.cpuAvg << " ms" << endl;
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
    return 0;
}