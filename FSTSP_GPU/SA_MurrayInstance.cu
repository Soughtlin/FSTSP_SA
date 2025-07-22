#include "read_MurrayInstance.h"
#include "metrics.h"
#include "SA_device.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


// Murray Instance
string MurrayInstance = "MurrayInstance";
vector<string> MurrayInstance_folders = {
        "20140810T123437v1",
        "20140810T123437v2",
        "20140810T123437v3",
        "20140810T123437v4",
        "20140810T123437v5",
        "20140810T123437v6",
        "20140810T123437v7",
        "20140810T123437v8",
        "20140810T123437v9",
        "20140810T123437v10",
        "20140810T123437v11",
        "20140810T123437v12",
        "20140810T123440v1",
        "20140810T123440v2",
        "20140810T123440v3",
        "20140810T123440v4",
        "20140810T123440v5",
        "20140810T123440v6",
        "20140810T123440v7",
        "20140810T123440v8",
        "20140810T123440v9",
        "20140810T123440v10",
        "20140810T123440v11",
        "20140810T123440v12",
        "20140810T123443v1",
        "20140810T123443v2",
        "20140810T123443v3",
        "20140810T123443v4",
        "20140810T123443v5",
        "20140810T123443v6",
        "20140810T123443v7",
        "20140810T123443v8",
        "20140810T123443v9",
        "20140810T123443v10",
        "20140810T123443v11",
        "20140810T123443v12",
};


int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const int R = 256;          // 并行线程数（可调）
    int threadsPerBlock = 128;
    int blocksPerGrid  = (R + threadsPerBlock -1)/threadsPerBlock;

    ofstream resultFile("result_MurrayInstance.txt");
    resultFile << left << setw(10) << "Instance"
               << setw(15) << "Obj"
               << setw(15) << "CPU Time(ms)" << "\n";

    int idxInst = 0;
    for(const auto& folder: MurrayInstance_folders){
        // ---------- 读取实例（仍在 CPU） ----------
        Instance instHost = readMurrayInstance(MurrayInstance + "/" + folder);
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