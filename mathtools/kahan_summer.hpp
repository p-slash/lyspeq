/* Specialized container to handle Fisher matrix and vector summations
 * using Kahan summation to reduce numerical errors.
 * These containers only hold the error, and the sum is externally stored in
 * the calling function. The add() function takes in a pointer to the input
 * array, the size of the array, a pointer to the sum array, and an offset for
 * where to start adding in the sum array.
*/
#ifndef KAHAN_SUMMER_H
#define KAHAN_SUMMER_H

#include <memory>
#include "core/omp_manager.hpp"

class KahanVectorSummer {
    std::unique_ptr<double[]> c;
    int N;
public:
    KahanVectorSummer(int size) : N(size) {
        c = std::make_unique<double[]>(N);
    };

    void add(const double *x, int M, double *sum, int offset) {
        double *sum_ptr = sum + offset, *c_ptr = c.get() + offset;
        for (int i = 0; i < M; ++i) {
            double y = x[i] - c_ptr[i];
            double t = sum_ptr[i] + y;
            c_ptr[i] = (t - sum_ptr[i]) - y;
            sum_ptr[i] = t;
        }
    }
};

class KahanVectorSummer2D {
    std::unique_ptr<double[]> c;
    int N;
public:
    KahanVectorSummer2D(int size) : N(size) {
        c = std::make_unique<double[]>(N * N);
    };

    /**
     * Adds the elements of the input array to the sum array using Kahan summation.
     * @param x MxM input array.
     * @param M Size of the input array.
     * @param sum Pointer to the NxN sum array.
     * @param offset Offset in the sum array where to start adding.
     */
    void add(const double *x, int M, double *sum, int offset=0) {
        double *sum_ptr = sum + offset, *c_ptr = c.get() + offset;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = i; j < M; ++j) {
                int idx = j + i * N;
                double y = x[j + i * M] - c_ptr[idx];
                double t = sum_ptr[idx] + y;
                c_ptr[idx] = (t - sum_ptr[idx]) - y;
                sum_ptr[idx] = t;
            }
        }
    }
};

#endif
