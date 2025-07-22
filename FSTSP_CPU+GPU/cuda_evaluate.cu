#include "cuda_evaluate.cuh"

#ifdef USE_CUDA

// 在 cuda_evaluate.h 中 extern 的定义
double *d_tauTruck = nullptr;
double *d_tauDrone = nullptr;

/**
 * 单解评估的 kernel：前缀和 + 并行归约无人机部分
 */
__global__ void evaluate_kernel(
    int c,
    const int   *seq,        // [c]
    const int   *serv,       // [c]
    const double *tauTruck,
    const double *tauDrone,
    double      *outObj      // single-element
) {
    extern __shared__ double sm[];     // 0..c 用于 truck 前缀, c+1..c+1+blockDim 用于归约
    double *truckT = sm;
    double *reduce = sm + (c+1);

    // 串行计算卡车到达时间前缀
    if (threadIdx.x == 0) {
        int pos = 0;
        double t = 0;
        for (int k = 0; k < c; ++k) {
            int cust = seq[k];
            t += tauTruck[pos*(c+2) + cust];
            pos = cust;
            truckT[k] = t;
        }
        // 回程时间放到 truckT[c]
        truckT[c] = t + tauTruck[pos*(c+2) + 0];
    }
    __syncthreads();

    // 并行归约无人机最晚返回时刻
    double localMax = 0.0;
    for (int k = threadIdx.x; k < c; k += blockDim.x) {
        int r = serv[k];
        if (r > 0) {
            int cust = seq[k];
            int rvIdx = k + r;
            int rvNode = (rvIdx < c ? seq[rvIdx] : 0);
            double tStart = truckT[k] + d_SL;
            double fly1   = tauDrone[cust*(c+2) + cust];
            double fly2   = tauDrone[cust*(c+2) + rvNode];
            localMax = fmax(localMax, tStart + fly1 + fly2 + d_SR);
        }
    }
    reduce[threadIdx.x] = localMax;
    __syncthreads();

    // 归约
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            reduce[threadIdx.x] = fmax(reduce[threadIdx.x],reduce[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double truckReturn = truckT[c];
        double droneMax    = reduce[0];
        outObj[0] = fmax(truckReturn, droneMax);
    }
}

/**
 * 把常量矩阵 & scalars 拷到 GPU
 */
void init_gpu_constants(const Instance &inst) {
    int c = inst.c;
    size_t dim     = size_t(c + 2);
    size_t matB    = dim * dim * sizeof(double);
    cudaError_t err;

    // 分配并拷贝 tauTruck
    err = cudaMalloc(&d_tauTruck, matB);
    assert(err == cudaSuccess);
    std::vector<double> flat(dim*dim);
    for (int i = 0; i < c+2; ++i)
        for (int j = 0; j < c+2; ++j)
            flat[i*dim + j] = inst.tauTruck[i][j];
    err = cudaMemcpy(d_tauTruck, flat.data(), matB, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // 分配并拷贝 tauDrone
    err = cudaMalloc(&d_tauDrone, matB);
    assert(err == cudaSuccess);
    for (int i = 0; i < c+2; ++i)
        for (int j = 0; j < c+2; ++j)
            flat[i*dim + j] = inst.tauDrone[i][j];
    err = cudaMemcpy(d_tauDrone, flat.data(), matB, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // 拷贝常量到 constant 内存
    err = cudaMemcpyToSymbol(d_SL, &inst.SL, sizeof(double));
    assert(err == cudaSuccess);
    err = cudaMemcpyToSymbol(d_SR, &inst.SR, sizeof(double));
    assert(err == cudaSuccess);
}

void free_gpu_constants() {
    if (d_tauTruck) cudaFree(d_tauTruck);
    if (d_tauDrone) cudaFree(d_tauDrone);
}

/**
 * 主机侧包装：拷入解、调 kernel、拷回结果
 */
double evaluate_gpu(const Instance &inst, const Solution &sol) {
    int c = inst.c;
    // 分配并拷贝 seq & service
    int *d_seq  = nullptr, *d_serv = nullptr;
    double *d_obj = nullptr;
    size_t isize = c * sizeof(int);
    cudaMalloc(&d_seq,  isize);
    cudaMalloc(&d_serv, isize);
    cudaMalloc(&d_obj,  sizeof(double));

    cudaMemcpy(d_seq,  sol.seq.data(),     isize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_serv, sol.service.data(), isize, cudaMemcpyHostToDevice);

    // launch kernel: 1 block, e.g. 256 线程，动态共享内存需要 (c+1 + 256)*sizeof(double)
    int threads = 16;
    size_t smem = (c+1 + threads) * sizeof(double);
    evaluate_kernel<<<1, threads, smem>>>(
        c, d_seq, d_serv, d_tauTruck, d_tauDrone, d_obj
    );
    cudaDeviceSynchronize();

    // 拷回结果
    double h_obj;
    cudaMemcpy(&h_obj, d_obj, sizeof(double), cudaMemcpyDeviceToHost);

    // 释放临时
    cudaFree(d_seq);
    cudaFree(d_serv);
    cudaFree(d_obj);

    return h_obj;
}

#endif