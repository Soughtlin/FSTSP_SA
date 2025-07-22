//
// Created by Song.H.L on 2025/7/5.
//

#include "SA.h"
#ifdef USE_CUDA
#include "cuda_evaluate.cuh"
static constexpr int GPU_EVAL_THRESHOLD = 5;
#endif

// ---------- 可行性修复 ----------
void repair(const Instance &inst, Solution &s) {
    int c = inst.c;
    for (int k = 0; k < c; ++k) {
        int r = s.service[k]; //service隐含了无人机(i,j,k)三元组的信息
        if (r == 0)
            continue;
        int launch = (k == 0 ? 0 : s.seq[k - 1]);
        int cust = s.seq[k];
        int rvIdx = k + r;
        int rvNode = (rvIdx < c ? s.seq[rvIdx] : 0);
        double d1 = inst.tauDrone[launch][cust];
        double d2 = inst.tauDrone[cust][rvNode];
        if (d1 + d2 + inst.SL + inst.SR > inst.droneEndurance)
            s.service[k] = 0; // 将无人机服务改为卡车服务
    }
}

// ---------- 初解 ----------
Solution genInitial(const Instance &inst) {
    Solution s;
    int c = inst.c;
    s.seq.resize(c);
    iota(s.seq.begin(), s.seq.end(), 1);
    shuffle(s.seq.begin(), s.seq.end(), rng);
    s.service.resize(c);
    for (int i = 0; i < c; ++i)
        s.service[i] = randInt(0, c - 1);
    repair(inst, s);
    return s;
}

// ---------- 评估 ----------
double evaluate_serial(const Instance &inst, const Solution &S) {
    int c = inst.c;
    double truckT = 0, droneT = 0;
    int pos = 0;
    for (int k = 0; k < c; ++k) {
        int cust = S.seq[k];
        truckT += inst.tauTruck[pos][cust];
        pos = cust;
        int r = S.service[k];
        if (r > 0) {
            int rvIdx = k + r;
            int rvNode = (rvIdx < c ? S.seq[rvIdx] : 0);
            double tStart = truckT + inst.SL;
            double fly1 = inst.tauDrone[pos][cust];   // pos==cust after truck move
            double fly2 = inst.tauDrone[cust][rvNode];
            double tEnd = tStart + fly1 + fly2 + inst.SR;
            droneT = max(droneT, tEnd);
        }
    }
    truckT += inst.tauTruck[pos][0];
    return max(truckT, droneT);
}

double evaluate(const Instance &inst, const Solution &S) {
#ifdef USE_CUDA
    if (inst.c >= GPU_EVAL_THRESHOLD) {
        // 当客户数达到阈值时用 GPU 加速
        return evaluate_gpu(inst, S);
    }
#endif
    return evaluate_serial(inst, S);
}


// ---------- 邻域算子 ----------
// 1) 交换算子（swapOp）——对顺序和服务类型并行交换
void swapOp(Solution &s) {
    int i = randInt(0, s.seq.size() - 1),
            j = randInt(0, s.seq.size() - 1);
    swap(s.seq[i], s.seq[j]);     // 客户访问顺序交换
    swap(s.service[i], s.service[j]); // 服务方式（卡车/无人机）同步交换
}

// 2) 插入算子（insertOp）——将一个客户摘出并插入到序列另一位置
void insertOp(Solution &s) {
    int n = s.seq.size();
    int i = randInt(0, n - 1),
            j = randInt(0, n - 1);
    if (i == j) return;              // 相同位置，无操作
    int node = s.seq[i], typ = s.service[i];
    s.seq.erase(s.seq.begin() + i);
    s.service.erase(s.service.begin() + i);
    if (i < j) --j;                  // 调整插入下标
    s.seq.insert(s.seq.begin() + j, node);
    s.service.insert(s.service.begin() + j, typ);
}

// 3) 反转算子（invertOp，也即2-opt）——对一段路径做整体反序
void invertOp(Solution &s) {
    int i = randInt(0, s.seq.size() - 1),
            j = randInt(0, s.seq.size() - 1);
    if (i > j) swap(i, j);           // 保证 i ≤ j
    reverse(s.seq.begin() + i, s.seq.begin() + j + 1);
    reverse(s.service.begin() + i, s.service.begin() + j + 1);
}

// 4) 服务方式切换算子（changeType）——随机改变某客户的服务载具
void changeType(Solution &s) {
    int i = randInt(0, s.seq.size() - 1);
    s.service[i] = randInt(0, s.seq.size() - 1);
}

// --------- 邻域候选解 ---------
Solution neighbor_4operator(const Instance &inst, const Solution &cur) {
    Solution nxt = cur;
    double r = rand01();
    if (r < 0.25)
        swapOp(nxt);
    else if (r < 0.5)
        insertOp(nxt);
    else if (r < 0.75)
        invertOp(nxt);
    else
        changeType(nxt);
    repair(inst, nxt);
    return nxt;
}

// ---------- Instance求解 ----------
RunStat solveInstance(const Instance &inst) {
    // 读取客户数
    int c = inst.c;
    // 每一温度层的迭代次数，与客户数成正比
    int iterPerTemp = N_ITER_UNIT * c;

    // ---------- 初始解生成 & 评估 ----------
    // 生成一个可行的初始解
    Solution best = genInitial(inst);
    // 计算初始解的目标值
    double bestObj = evaluate(inst, best);
    Solution cur = best;
    double curObj = bestObj;

    // ---------- SA 参数初始化 ----------
    double T = T0; // 初始温度
    int R = 0;
    auto t0 = Clock::now();

    // ---------- 主循环：直到非改进次数达到阈值 ----------
    while (R < N_NON_IMPROVING) {
        bool improve = false;

        // ---------- 内层循环：在当前温度下做多次邻域搜索 ----------
        for (int it = 0; it < iterPerTemp; ++it) {
            // 在当前解基础上生成一个邻域候选解
            Solution cand;
            if (c <= 10) {
                cand = neighbor_4operator(inst, cur);
            } else {
                cand = neighbor_6operator(inst, cur); // 客户更多时引入更多邻近算子
            }
            double obj = evaluate(inst, cand);
            double diff = obj - curObj;

            // Metropolis 接受准则: 更优则无条件接受
            if (diff <= 0 || rand01() < exp(-diff / T)) {
                cur = cand; // 接受候选解，更新当前解
                curObj = obj;

                // 如果比历史最优更好，则更新全局最优
                if (curObj < bestObj) {
                    best = cur;
                    bestObj = curObj;
                    improve = true;
                }
            }
        }

        // ---------- 更新非改进计数 & 降温 ----------
        if (improve) {
            R = 0; // 若本温度层有改进，重置非改进计数
        } else {
            ++R; // 否则非改进计数加一
        }
        T *= ALPHA; // 按比例因子降温
    }

    // ---------- 收尾：记录结束时间 & 输出 ----------
    auto t1 = Clock::now();
    double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    return {bestObj, ms};
}

// 5) Or-opt：摘取长度为 k 的连续客户段并插入到另一位置
void orOptOp(Solution &s, int k) {
    int n = s.seq.size();
    if (k <= 0 || k >= n) return;
    // 随机选段起点 i
    int i = randInt(0, n - k);
    // 随机选段插入位置 j（防止插到自己段内）
    int j = randInt(0, n - k);
    if (j >= i && j <= i + k) return;
    // 把段复制
    vector<int> segSeq(s.seq.begin() + i, s.seq.begin() + i + k);
    vector<int> segServ(s.service.begin() + i, s.service.begin() + i + k);
    // 删除原段
    s.seq.erase(s.seq.begin() + i, s.seq.begin() + i + k);
    s.service.erase(s.service.begin() + i, s.service.begin() + i + k);
    // 调整插入下标
    if (j > i) j -= k;
    // 插回去
    s.seq.insert(s.seq.begin() + j, segSeq.begin(), segSeq.end());
    s.service.insert(s.service.begin() + j, segServ.begin(), segServ.end());
}

// 6) 子序列交叉交换（cross-exchange）：随机交换两段（长度可不同）
void crossExchangeOp(Solution &s) {
    int n = s.seq.size();
    // 随机选两段长度（这里上限设为 3，可调）
    int len1 = randInt(1, min(3, n / 2));
    int len2 = randInt(1, min(3, n / 2));
    int i = randInt(0, n - len1), j = randInt(0, n - len2);
    // 只在两段不重叠时执行
    if ((i + len1 <= j) || (j + len2 <= i)) {
        // 复制两段
        vector<int> s1_seq(s.seq.begin() + i, s.seq.begin() + i + len1),
                s1_serv(s.service.begin() + i, s.service.begin() + i + len1);
        vector<int> s2_seq(s.seq.begin() + j, s.seq.begin() + j + len2),
                s2_serv(s.service.begin() + j, s.service.begin() + j + len2);
        // 先删除后面段，再删除前面段
        if (i < j) {
            s.seq.erase(s.seq.begin() + j, s.seq.begin() + j + len2);
            s.service.erase(s.service.begin() + j, s.service.begin() + j + len2);
            s.seq.erase(s.seq.begin() + i, s.seq.begin() + i + len1);
            s.service.erase(s.service.begin() + i, s.service.begin() + i + len1);
            // 插入对方段
            s.seq.insert(s.seq.begin() + i, s2_seq.begin(), s2_seq.end());
            s.service.insert(s.service.begin() + i, s2_serv.begin(), s2_serv.end());
            s.seq.insert(s.seq.begin() + j - len1 + len2, s1_seq.begin(), s1_seq.end());
            s.service.insert(s.service.begin() + j - len1 + len2, s1_serv.begin(), s1_serv.end());
        } else {
            // i>j 时对称处理
            s.seq.erase(s.seq.begin() + i, s.seq.begin() + i + len1);
            s.service.erase(s.service.begin() + i, s.service.begin() + i + len1);
            s.seq.erase(s.seq.begin() + j, s.seq.begin() + j + len2);
            s.service.erase(s.service.begin() + j, s.service.begin() + j + len2);
            s.seq.insert(s.seq.begin() + j, s1_seq.begin(), s1_seq.end());
            s.service.insert(s.service.begin() + j, s1_serv.begin(), s1_serv.end());
            s.seq.insert(s.seq.begin() + i - len2 + len1, s2_seq.begin(), s2_seq.end());
            s.service.insert(s.service.begin() + i - len2 + len1, s2_serv.begin(), s2_serv.end());
        }
    }
}

// --------- 邻域候选解 (引入两种更强的邻域算子) ---------
Solution neighbor_6operator(const Instance &inst, const Solution &cur) {
    Solution nxt = cur;
    double r = rand01();
    if (r < 0.17) swapOp(nxt);
    else if (r < 0.34) insertOp(nxt);
    else if (r < 0.51) invertOp(nxt);
    else if (r < 0.68) changeType(nxt);
    else if (r < 0.84) orOptOp(nxt, 2);      // Or-opt 段长 k=2
    else crossExchangeOp(nxt);
    repair(inst, nxt);
    return nxt;
}

