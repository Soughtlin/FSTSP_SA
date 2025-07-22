#ifndef SRC_INSTANCE_H
#define SRC_INSTANCE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <regex>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>

using namespace std;
using Clock = chrono::high_resolution_clock;   // 精确计时

// ============ 随机数 ============
static random_device rd;  // 用硬件熵播种
static mt19937 rng(rd());

double rand01();

int randInt(int a, int b);


// ============ 数据结构 ============
struct Instance {
    int c = 0;             // 客户数
    double SL = 1.0;           // 起飞准备
    double SR = 1.0;           // 回收耗时
    double droneEndurance = 0.0;           // 续航 E
    vector<vector<double>> tauTruck;          // (c+2)×(c+2) 行驶时间
    vector<vector<double>> tauDrone;          // (c+2)×(c+2) 飞行时间
    double BKS = numeric_limits<double>::quiet_NaN(); // 最佳已知
    string name;                               // 文件名(或 id)
};

struct RunStat {
    double obj = 0.0;        // 完工时间 (makespan)
    double millis = 0.0;        // 耗时 (ms)
};
struct InstStat {
    Instance inst;
    vector<RunStat> runs;             // size=R
};


// ============ 解编码 ============
struct Solution {
    vector<int> seq;       // 访问顺序 (c)
    vector<int> service;   // 0..c-1
};

// ============ SA 参数 ============
static const double T0 = 2.0;   // 初始温度
static const double ALPHA = 0.975; // 降温系数
static const int N_ITER_UNIT = 2000; // 同温度最大迭代次数
static const int N_NON_IMPROVING = 20; // 无改进最大迭代次数 -> 收敛


#endif //SRC_INSTANCE_H
