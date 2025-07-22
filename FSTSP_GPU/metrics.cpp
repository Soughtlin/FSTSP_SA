#include "metrics.h"

// ============ 指标计算 ============
Metrics computeMetrics(const InstStat &stat, int R) {
    Metrics m;

    double sumRun = 0;
    double bestRun = 1e100;
    double sumCpuOne = 0, maxCpuOne = 0;

    // 计算当前算例的最优解和平均解
    for (const auto &r: stat.runs) {
        bestRun = min(bestRun, r.obj);  // 最优解
        sumRun += r.obj;  // 所有运行的目标值
        sumCpuOne += r.millis;  // 计算当前算例的总CPU时间
        maxCpuOne = max(maxCpuOne, r.millis);  // 当前算例的最大CPU时间
    }

    // 计算目标函数值的平均值
    double avgRun = sumRun / R;

    // 存储计算结果
    m.obj_best = bestRun;  // 最优解
    m.obj_avg = avgRun;    // 平均目标函数值
    m.cpuAvg = sumCpuOne / R;  // 平均运行时间
    m.cpuMax = maxCpuOne;  // 最大运行时间

    return m;
}