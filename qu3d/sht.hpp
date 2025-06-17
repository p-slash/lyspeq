#include <algorithm>
#include "sht.h"

class SHT
{
    long lmax, nspace, quarter_nspace;
    int pm_order;

public:
    SPHERE_MODES Alm;
    DOUBLE_MAP map;

    SHT(long ellmax, int pm_o, long nsp)
            : lmax(ellmax), pm_order(pm_o), nspace(nsp)
    {
        quarter_nspace = nspace / 4;

        allocate_sphere_modes(&Alm, lmax);
        allocate_double_map(
            &map, -pm_order - 1,
            nspace + pm_order + 1,
            -quarter_nspace - pm_order - 1,
            quarter_nspace + pm_order + 1,
            0);
    }

    ~SHT() {
        deallocate_sphere_modes(&Alm);
        deallocate_double_map(&map);
    }
    
    void zeroAlm() {
        std::fill_n(Alm.vector, Alm.Nmode, 0);
    }

    void analysis() {
        sht_grid_analysis_1(
            Alm.coefs, map.matrix, lmax, nspace,
            map.xmin, map.xmax, map.ymin, map.ymax,
            nullptr, 0);
    }

    void synthesis() {
        sht_grid_synthesis_1(
            Alm.coefs, map.matrix, lmax, nspace,
            map.xmin, map.xmax, map.ymin, map.ymax,
            nullptr, 0);
    }

    double interpolate(double theta, double phi) const {
        double result = 0;
        forward_sphere_pm_1(
            map.matrix, &theta, &phi, &result, pm_order, 1, quarter_nspace);
        return result;
    }

    void reverseInterpolate(double theta, double phi, double val) {
        reverse_sphere_pm_1(
            map.matrix, &theta, &phi, &val, pm_order, 1, quarter_nspace);
    }
};
