#include "instance.h"

thread_local std::mt19937 rng;

double rand01() {
    static uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

int randInt(int a, int b) {
    uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

void seed_rng(unsigned long long seed) {
    rng.seed(seed);
}
