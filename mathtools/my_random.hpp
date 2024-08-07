#ifndef MY_RANDOM_H
#define MY_RANDOM_H

#include <random>

class MyRNG {
public:
    MyRNG() :uidist(0, 1) {};

    void seed(size_t in) { rng_engine.seed(in); }

    double normal() { return n_dist(rng_engine); }
    void fillVectorNormal(double *v, unsigned int size) {
        for (unsigned int i = 0; i < size; ++i)
            v[i] = n_dist(rng_engine);
    }

    void fillVectorOnes(double *v, size_t size) {
        for (size_t i = 0; i < size; ++i)
            v[i] = (uidist(rng_engine) == 0) ? -1 : +1;
    }

private:
    std::mt19937_64 rng_engine;
    std::normal_distribution<double> n_dist;
    std::uniform_int_distribution<int> uidist;
};

#endif
