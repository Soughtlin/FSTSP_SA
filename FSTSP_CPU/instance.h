#ifndef SRC_INSTANCE_H
#define SRC_INSTANCE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <regex>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>
#include <cassert>

using namespace std;
using Clock = chrono::high_resolution_clock;   // 精确计时

// 线程安全随机数，后面要为每个线程设置随机种子
extern thread_local std::mt19937 rng;
void seed_rng(unsigned long long fseed);

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
    RunStat() : obj(0.0), millis(0.0) {}
    RunStat(double _obj, double _millis) : obj(_obj), millis(_millis) {}
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
static const int N_NON_IMPROVING = 5; // 无改进最大迭代次数 -> 收敛


#endif //SRC_INSTANCE_H
