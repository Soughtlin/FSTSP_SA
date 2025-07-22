//
// Created by Song.H.L on 2025/7/6.
//

#ifndef SRC_MPI_UTILS_H
#define SRC_MPI_UTILS_H

# include "instance.h"
# include <mpi.h>

void broadcastInstance(InstStat &st, int rank, int root);

void gatherRunStats(const std::vector<RunStat> &localRuns, int totalR, int rank, int size,
                    std::vector<RunStat> &allRuns);

#endif //SRC_MPI_UTILS_H
