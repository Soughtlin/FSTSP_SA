#pragma once

#include "instance.h"


#ifdef USE_CUDA

#ifdef __CUDACC__
// 只有在 nvcc 编译时才会看到这些声明
__constant__ double d_SL;
__constant__ double d_SR;
//__device__   double *d_tauTruck;
//__device__   double *d_tauDrone;
#else
// 主机代码只看到 extern 声明
extern double d_SL;
extern double d_SR;
extern double *d_tauTruck;
extern double *d_tauDrone;
#endif

// 初始化 / 释放 GPU 常量
void init_gpu_constants(const Instance &inst);
void free_gpu_constants();

// GPU 评估函数
double evaluate_gpu(const Instance &inst, const Solution &sol);

#else  // 如果不使用 CUDA

// 当没有启用 CUDA 时，这些函数为空实现，避免冲突
inline void init_gpu_constants(const Instance &) {}
inline void free_gpu_constants() {}
inline double evaluate_gpu(const Instance &, const Solution &) { return 0.0; }

#endif  // USE_CUDA
