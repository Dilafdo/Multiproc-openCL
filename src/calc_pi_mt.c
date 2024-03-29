//
// Created by ruksh on 16/03/2024.
//

#include <omp.h>
#include <logger.h>

#define NUM_THREADS PROCESSOR_COUNT*2
#define PAD 4 // assume 64 byte L1 cache line size

double calc_pi(int num_steps) {
    int i;
    double pi, x, sum = 0.0;

    const double step = 1.0 / (double) num_steps;

    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    return pi;
}

double calc_pi_mt(int num_steps) {
    double pi = 0.0;

    const double step = 1.0 / (double) num_steps;
    int nThreads = NUM_THREADS;
    logger("Initial thread count prediction : %d", nThreads);

    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel
    {
        int i, id, nLocalThreads;
        double x, sum;

        id = omp_get_thread_num();
        nLocalThreads = omp_get_num_threads();
        if (id == 0) {
            nThreads = nLocalThreads;
        }
        for (i = id, sum = 0.0; i < num_steps; i = i + nThreads) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp critical
        pi += sum * step;
    }
    return pi;
}

double calc_pi_mt_false_sharing(int num_steps) {
    double pi;

    const double step = 1.0 / (double) num_steps;
    double sums[NUM_THREADS][PAD];
    int nThreads = NUM_THREADS;
    logger("Initial thread count prediction : %d", nThreads);

    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel
    {
        int i, id, nLocalThreads;
        double x;

        id = omp_get_thread_num();
        nLocalThreads = omp_get_num_threads();
        if (id == 0) {
            nThreads = nLocalThreads;
        }
        for (i = id, sums[id][0] = 0.0; i < num_steps; i = i + nThreads) {
            x = (i + 0.5) * step;
            sums[id][0] += 4.0 / (1.0 + x * x);
        }
    }
    int i;
    for (i = 0, pi = 0.0; i < nThreads; i++) {
        pi += sums[i][0] * step;
    }
    return pi;
}
