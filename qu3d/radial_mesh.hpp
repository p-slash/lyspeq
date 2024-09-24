#ifndef RADIAL_MESH_H
#define RADIAL_MESH_H

#include <memory>

#include "qu3d/sht.h"


class SphereModes {
    SPHERE_MODES alm;
public:
    int lmax;
    SphereModes(int lmax_) : lmax(lmax_) {
        allocate_sphere_modes(&alm, lmax);
        zeroVector();
    }
    ~SphereModes() { deallocate_sphere_modes(&alm); }

    SPHERE_MODES* get() { return &alm; }
    void zeroVector() { std::fill_n(alm.vector, alm.Nmode, 0); }
}


class RadialMesh {
    std::vector<std::unique_ptr<SphereModes>> alm_slices;
    std::unique_ptr<double[]> theta, phi;
public:
    RadialMesh(int lmax_, int nrbins) {
        alm_slices.reserve(nrbins);
        for (int i = 0; i < nrbins; ++i)
            alm_slices.push_back(std::make_unique<SphereModes>(lmax_));
    }
};

#endif
