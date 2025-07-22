#ifndef SRC_SA_GPU_CUH
#define SRC_SA_GPU_CUH

#include "instance.h"
#include <curand_kernel.h>


// 在设备端使用的随机数工具
__device__ inline double rand01(curandState *state) {
    return curand_uniform_double(state);
}

__device__ inline int randInt(curandState *state, int a, int b) {
    return a + static_cast<int>(curand_uniform_double(state) * (b - a + 1));
}

struct DeviceRunStat {
    double obj;
    double millis;
};

// ---------- 内核调用入口 ----------
__global__ void SA_kernel(DeviceInstance d_inst,int R,DeviceRunStat *d_out);

#endif //SRC_SA_GPU_CUH
