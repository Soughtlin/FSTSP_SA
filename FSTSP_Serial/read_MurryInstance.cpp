#include "read_MurryInstance.h"

// 读取无索引的纯 CSV 矩阵（以逗号分隔）
vector<vector<double>> readCSVMatrix(const string &fname) {
    ifstream in(fname);
    if (!in.is_open())
        throw runtime_error("无法打开文件: " + fname);

    vector<vector<double>> M;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        M.push_back(row);
    }
    in.close();
    return M;
}

Instance readMurrayInstance(const string &path) {
    Instance inst;
    inst.name = path;

    // 1. 读 nodes.csv，只为得到节点总数 n
    vector<int> ids;
    {
        ifstream fn(path + "/nodes.csv");
        if (!fn.is_open())
            throw runtime_error("无法打开 " + path + "/nodes.csv");
        string line;
        while (getline(fn, line)) {
            if (line.empty()) continue;
            stringstream ss(line);
            string f;
            vector<string> F;
            while (getline(ss, f, ',')) F.push_back(f);
            // F = { id, x, y, drone_flag }
            ids.push_back(stoi(F[0]));
        }
        fn.close();
    }
    // Murray&Chu 的格式：总节点数 = c + 2
    int n = (int) ids.size();
    if (n < 2)
        throw runtime_error("节点数量不合法: " + to_string(n));
    inst.c = n - 2;

    // 2. 读卡车时间矩阵 tau.csv
    inst.tauTruck = readCSVMatrix(path + "/tau.csv");
    if ((int) inst.tauTruck.size() != n)
        throw runtime_error("tau.csv 大小不符，期望 " + to_string(n) + "×" + to_string(n));

    // 3. 读无人机时间矩阵 tauprime.csv
    inst.tauDrone = readCSVMatrix(path + "/tauprime.csv");
    if ((int) inst.tauDrone.size() != n)
        throw runtime_error("tauprime.csv 大小不符，期望 " + to_string(n) + "×" + to_string(n));

    // 4. （可选）如果你有一个外部参数文件或者固定值，可以在这里赋给 SL/SR/E
    // inst.SL = ...;
    // inst.SR = ...;
    // inst.droneEndurance = ...;
    // inst.BKS = ...;  // 如果有最佳已知值

    return inst;
}