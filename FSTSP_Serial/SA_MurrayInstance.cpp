# include "SA.h"
#include "read_MurryInstance.h"
#include "metrics.h"


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


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string instance = MurrayInstance;
    vector<string> folders = MurrayInstance_folders;
    const int R = 1;

    vector<Metrics> metrics;
    ofstream resultFile("result_MurrayInstance.txt");

    // 给每列预留固定宽度，并左对齐
    resultFile << left
               << setw(10) << "Instance"
               << setw(15) << "Obj"
               << setw(15) << "CPU Time(ms)"
               << "\n";

    int i = 0;
    for (auto &folder: folders) {
        InstStat st;
        st.inst = readMurrayInstance(instance + "/" + folder);

        // ---- 求解 ----
        cerr << "[info] solving " << st.inst.name
             << " (c=" << st.inst.c << ")"
             << " Instance " << ++i << "\n";
        for (int r = 0; r < R; ++r) {
            RunStat rs = solveInstance(st.inst);
            st.runs.push_back(rs);
        }

        // ---- 计算指标 ----
        Metrics M = computeMetrics(st, R);
        metrics.push_back(M);  // 把结果写回已分配好的 vector

        cout << fixed << setprecision(4);
        cout << "obj best = " << M.obj_best << "\n";
        cout << "CPU time Avg = " << M.cpuAvg << " ms" << endl;
    }

    // 最后统一写文件，每列宽度和表头保持一致
    resultFile << fixed << setprecision(4);
    for (int j = 0; j < metrics.size(); ++j) {
        resultFile << left
                   << setw(10) << (j+1)
                   << setw(15) << metrics[j].obj_best
                   << setw(15) << metrics[j].cpuAvg
                   << "\n";
    }

    resultFile.close();
    return 0;
}