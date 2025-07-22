#ifndef SRC_METRICS_H
#define SRC_METRICS_H

#include "instance.h"

struct Metrics {
    double obj_best = 1e10;
    double obj_avg = 1e10;
    double cpuAvg = 0;
    double cpuMax = 0;
};

Metrics computeMetrics(const InstStat &stat, int R);


#endif //SRC_METRICS_H
