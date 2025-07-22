#include "SA_device.cuh"
#include <math_constants.h>

#define N_ITER_UNIT 2000
#define N_NON_IMPROVING 20
#define T0 2.0
#define ALPHA 0.975

#ifndef MAX_C
#define MAX_C 256
#endif

// ---------- repair ----------
__device__ void repair(const DeviceInstance &inst, int *seq, int *service) {
    int c = inst.c;
    int n = c + 2;
    for (int k = 0; k < c; ++k) {
        int r = service[k];
        if (r == 0) continue;
        int launch = (k == 0 ? 0 : seq[k - 1]);
        int cust = seq[k];
        int rvIdx = k + r;
        int rvNode = (rvIdx < c ? seq[rvIdx] : 0);
        double d1 = inst.tauDrone[launch * n + cust];
        double d2 = inst.tauDrone[cust * n + rvNode];
        if (d1 + d2 + inst.SL + inst.SR > inst.droneEndurance)
            service[k] = 0;
    }
}

// ---------- genInitial ----------
__device__ void shuffleSeq(int *seq, int n, curandState *state) {
    for (int i = n - 1; i > 0; --i) {
        int j = randInt(state, 0, i);
        int tmp = seq[i];
        seq[i] = seq[j];
        seq[j] = tmp;
    }
}

__device__ void genInitial(const DeviceInstance &inst, int *seq, int *service,
                           curandState *state) {
    int c = inst.c;
    for (int i = 0; i < c; ++i) seq[i] = i + 1;
    shuffleSeq(seq, c, state);
    for (int i = 0; i < c; ++i) service[i] = randInt(state, 0, c - 1);
    repair(inst, seq, service);
}

// ---------- evaluate ----------
__device__ double evaluate(const DeviceInstance &inst,
                           const int *seq, const int *service) {
    int c = inst.c;
    int n = c + 2;
    double truckT = 0, droneT = 0;
    int pos = 0;
    for (int k = 0; k < c; ++k) {
        int cust = seq[k];
        truckT += inst.tauTruck[pos * n + cust];
        pos = cust;
        int r = service[k];
        if (r > 0) {
            int rvIdx = k + r;
            int rvNode = (rvIdx < c ? seq[rvIdx] : 0);
            double tStart = truckT + inst.SL;
            double fly1 = inst.tauDrone[pos * n + cust];
            double fly2 = inst.tauDrone[cust * n + rvNode];
            double tEnd = tStart + fly1 + fly2 + inst.SR;
            if (tEnd > droneT) droneT = tEnd;
        }
    }
    truckT += inst.tauTruck[pos * n + 0];
    return truckT > droneT ? truckT : droneT;
}

// ---------- 4 operators (仅演示 swap / insert 两个，其余同理) ----------
__device__ void swapOp(int *seq, int *service, int n, curandState *state) {
    int i = randInt(state, 0, n - 1), j = randInt(state, 0, n - 1);
    int t = seq[i];
    seq[i] = seq[j];
    seq[j] = t;
    t = service[i];
    service[i] = service[j];
    service[j] = t;
}

__device__ void insertOp(int *seq, int *service, int n, curandState *state) {
    int i = randInt(state, 0, n - 1), j = randInt(state, 0, n - 1);
    if (i == j) return;
    int node = seq[i], typ = service[i];
    if (i < j) {
        for (int k = i; k < j; ++k) {
            seq[k] = seq[k + 1];
            service[k] = service[k + 1];
        }
        j--;
    } else {
        for (int k = i; k > j; --k) {
            seq[k] = seq[k - 1];
            service[k] = service[k - 1];
        }
    }
    seq[j] = node;
    service[j] = typ;
}
// (invertOp, changeType, etc. 直接从 SA.cpp 改 __device__，此处略)

// ---------- neighbor ----------
__device__ void neighbor_4operator(const DeviceInstance &inst,
                                   int *seq, int *service,
                                   curandState *state) {
    double r = rand01(state);
    int c = inst.c;
    if (r < 0.25) swapOp(seq, service, c, state);
    else if (r < 0.5) insertOp(seq, service, c, state);
    // else if... invertOp / changeType 同理
    repair(inst, seq, service);
}

// ---------- solveInstance_device ----------
__device__ DeviceRunStat solveInstance_device(const DeviceInstance &inst, curandState *rngState) {
    const int c = inst.c;
    if(c>MAX_C){
        printf("Error: MAX_C exceeded\n");
        return {1e5,0.0};
    }
    const int iterPerTemp = N_ITER_UNIT * c;

    // per-thread local arrays (最多 c≤??，可 static assert)
    int seq_max[MAX_C];   // 假设 c ≤ 128
    int serv_max[MAX_C];
    int curSeq[MAX_C], curServ[MAX_C];

    genInitial(inst, seq_max, serv_max, rngState);
    double bestObj = evaluate(inst, seq_max, serv_max);
    for (int k = 0; k < c; ++k) {
        curSeq[k] = seq_max[k];
        curServ[k] = serv_max[k];
    }
    double curObj = bestObj;

    double T = T0;
    int R = 0;
    while (R < N_NON_IMPROVING) {
        bool improve = false;
        for (int it = 0; it < iterPerTemp; ++it) {
            int candSeq[MAX_C], candServ[MAX_C];
            for (int k = 0; k < c; ++k) {
                candSeq[k] = curSeq[k];
                candServ[k] = curServ[k];
            }
            neighbor_4operator(inst, candSeq, candServ, rngState);
            double obj = evaluate(inst, candSeq, candServ);
            double diff = obj - curObj;
            if (diff <= 0 || rand01(rngState) < exp(-diff / T)) {
                for (int k = 0; k < c; ++k) {
                    curSeq[k] = candSeq[k];
                    curServ[k] = candServ[k];
                }
                curObj = obj;
                if (curObj < bestObj) {
                    for (int k = 0; k < c; ++k) {
                        seq_max[k] = curSeq[k];
                        serv_max[k] = curServ[k];
                    }
                    bestObj = curObj;
                    improve = true;
                }
            }
        }
        R = improve ? 0 : R + 1;
        T *= ALPHA;
    }

    DeviceRunStat rs{bestObj, 0.0};
    return rs;
}

__global__ void SA_kernel(DeviceInstance d_inst,int R,DeviceRunStat *d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R) return;

    // CUDA RNG 初始化
    curandState state;
    curand_init(1234ULL + idx, 0, 0, &state);

    DeviceRunStat rs = solveInstance_device(d_inst, &state);

    d_out[idx] = rs;
}