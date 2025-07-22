#include "read_PonzaInstance.h"
#include "metrics.h"
#include "SA_device.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// —— 读取 .dat 文件 ——
string PonzaInstance = "PonzaInstances";
const vector<string> PonzaInstance_files = {
        "Instance_005.1_new.dat",
        "Instance_005.2_new.dat",
        "Instance_005.3_new.dat",
        "Instance_005.4_new.dat",
        "Instance_005.5_new.dat",
        "Instance_006.1_new.dat",
        "Instance_006.2_new.dat",
        "Instance_006.3_new.dat",
        "Instance_006.4_new.dat",
        "Instance_006.5_new.dat",
        "Instance_007.1_new.dat",
        "Instance_007.2_new.dat",
        "Instance_007.3_new.dat",
        "Instance_007.4_new.dat",
        "Instance_007.5_new.dat",
        "Instance_008.1_new.dat",
        "Instance_008.2_new.dat",
        "Instance_008.3_new.dat",
        "Instance_008.4_new.dat",
        "Instance_008.5_new.dat",
        "Instance_009.1_new.dat",
        "Instance_009.2_new.dat",
        "Instance_009.3_new.dat",
        "Instance_009.4_new.dat",
        "Instance_009.5_new.dat",
        "Instance_010.1_new.dat",
        "Instance_010.2_new.dat",
        "Instance_010.3_new.dat",
        "Instance_010.4_new.dat",
        "Instance_010.5_new.dat",
        "Instance_050.1_new.dat",
        "Instance_050.2_new.dat",
        "Instance_050.3_new.dat",
        "Instance_050.4_new.dat",
        "Instance_050.5_new.dat",
        "Instance_100.1_new.dat",
        "Instance_100.2_new.dat",
        "Instance_100.3_new.dat",
        "Instance_100.4_new.dat",
        "Instance_100.5_new.dat",
        "Instance_150.1_new.dat",
        "Instance_150.2_new.dat",
        "Instance_150.3_new.dat",
        "Instance_150.4_new.dat",
        "Instance_150.5_new.dat",
        "Instance_200.1_new.dat",
        "Instance_200.2_new.dat",
        "Instance_200.3_new.dat",
        "Instance_200.4_new.dat",
        "Instance_200.5_new.dat"
};


int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const int R = 10;          // 并行线程数（可调）
    int threadsPerBlock = 128;
    int blocksPerGrid  = (R + threadsPerBlock -1)/threadsPerBlock;

    ofstream resultFile("result_PonzaInstance.txt");
    resultFile << left << setw(10) << "Instance"
               << setw(15) << "Obj"
               << setw(15) << "CPU Time(ms)" << "\n";

    int idxInst = 0;
    for(const auto& file: PonzaInstance_files){
        // ---------- 读取实例（仍在 CPU） ----------
        Instance instHost = readPonzaInstance(PonzaInstance + "/" + file);
        cerr<<"[info] solving "<<instHost.name<<" (c="<<instHost.c<<")\n";

        // ---------- 打平成 DeviceInstance ----------
        DeviceInstance instDev {};
        std::vector<double> h_tauT, h_tauD;
        flattenInstance(instHost, instDev, h_tauT, h_tauD);

        const int n = instHost.c + 2;
        const size_t bytes = sizeof(double)*n*n;

        // unified memory
        cudaMallocManaged(&instDev.tauTruck, bytes);
        cudaMallocManaged(&instDev.tauDrone, bytes);
        memcpy(instDev.tauTruck, h_tauT.data(), bytes);
        memcpy(instDev.tauDrone, h_tauD.data(), bytes);

        // ---------- 结果数组 ----------
        DeviceRunStat* d_res;
        cudaMallocManaged(&d_res, sizeof(DeviceRunStat)*R);

        // ---------- 启动 kernel ----------
        cudaEvent_t start, end;
        float elapsedTime;
        cudaEventCreate(&start); // 创建 CUDA 事件
        cudaEventCreate(&end);
        cudaEventRecord(start); // 记录开始时间
        SA_kernel<<<blocksPerGrid, threadsPerBlock>>>(instDev, R, d_res);
        cudaEventRecord(end); // 记录结束时间
        cudaEventSynchronize(end);  // 等待直到 stop 完全记录
        cudaEventElapsedTime(&elapsedTime, start, end);  //ms
        cudaEventDestroy(start); // 销毁 CUDA 事件
        cudaEventDestroy(end);


        // ---------- 汇总 ----------
        RunStat agg{};
        agg.obj = 1e100;
        agg.millis = 0.0;
        for(int i=0;i<R;++i){
            agg.obj = min(agg.obj, d_res[i].obj);
            agg.millis += d_res[i].millis;
        }
        agg.millis /= R;

        cout<<fixed<<setprecision(4)
            <<"obj best = "<<agg.obj<<"\n"
            <<"CPU time Avg = "<<elapsedTime<<" ms\n";

        resultFile<<left<<setw(10)<<(++idxInst)
                  <<setw(15)<<agg.obj
                  <<setw(15)<<elapsedTime<<"\n";

        // ---------- 释放 ----------
        cudaFree(instDev.tauTruck);
        cudaFree(instDev.tauDrone);
        cudaFree(d_res);
    }
    resultFile.close();
    return 0;
}