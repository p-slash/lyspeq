#include "mathtools/smoother.hpp"
#include "mathtools/stats.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each
#include <numeric>  // std::accumulate
#include <vector>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

std::unique_ptr<Smoother> process::smoother;


void padArray(
        const double *arr, int size, int hwsize, std::vector<double> &out
) {
    out.resize(size + 2 * hwsize, 0);

    std::copy_n(arr, size, out.begin() + hwsize);
    std::fill_n(out.begin(), hwsize, arr[0]);
    std::fill_n(out.begin() + size, hwsize, arr[size - 1]);
}


double _getMedianBelowThreshold(
        double *sorted_arr, int size, int &newsize, double thres=1e3
) {
    newsize = size;
    while ((newsize != 0) && sorted_arr[newsize - 1] > thres)
        --newsize;

    if (newsize == 0)
        return sorted_arr[0];

    return stats::medianOfSortedArray(sorted_arr, newsize);
}


void _findMedianStatistics(
        double *arr, int size, double &median, double &mean, double &mad,
        double thres=1e3
) {
    int newsize;
    std::sort(arr, arr + size);
    median = _getMedianBelowThreshold(arr, size, newsize, thres);
    mean = std::accumulate(arr, arr + newsize, 0.) / newsize;

    std::for_each(arr, arr + newsize, [median](double &f) { f = fabs(f - median); });
    std::sort(arr, arr + newsize);

    // The constant factor makes it unbiased
    mad = 1.4826 * stats::medianOfSortedArray(arr, newsize);
}


Smoother::Smoother(ConfigFile &config)
{
    LOG::LOGGER.STD("###############################################\n");
    LOG::LOGGER.STD("Reading noise smoothing parameters from config.\n");

    config.addDefaults(smoother_default_parameters);
    sigmapix = config.getInteger("SmoothNoiseWeights");
    LOG::LOGGER.STD("SmoothNoiseWeights is set to %d.\n\n", sigmapix);
 
    is_smoothing_on = (sigmapix >= 0);
    use_mean = (sigmapix == 0);
    is_smoothing_on_rmat =
        (config.getInteger("SmoothResolutionMatrix") > 0) && is_smoothing_on;

    if (!is_smoothing_on || use_mean)
        return;

    double sum = 0, *g = &gaussian_kernel[0];
    for (int i = -HWSIZE; i < HWSIZE + 1; ++i, ++g) {
        *g = exp(-pow(i * 1. / sigmapix, 2) / 2);
        sum += *g;
    }

    std::for_each(
        &gaussian_kernel[0], &gaussian_kernel[0] + KS, 
        [sum](double &x) { x /= sum; });
}


void Smoother::smooth1D(double *inplace, int size, int ndim) {
    std::vector<double> tempvector(size + 2 * HWSIZE, 0);
    double median __attribute__((unused)), mad __attribute__((unused)),
           mean;

    std::copy_n(inplace, size, tempvector.begin());
    _findMedianStatistics(tempvector.data(), size, median, mean, mad);

    if (use_mean) {
        std::fill_n(inplace, size, mean);
    } else {
        padArray(inplace, size, HWSIZE, tempvector);

        // Convolve
        for (int i = 0; i < size; ++i)
            inplace[i] = cblas_ddot(KS, gaussian_kernel, 1, tempvector.data() + i, 1);
    }

    if (ndim > 1)
        Smoother::smooth1D(inplace + size, size, ndim - 1);
}


void Smoother::smoothNoise(const double *n2, double *out, int size) {
    double median, mad, mean;

    std::copy_n(n2, size, out);
    _findMedianStatistics(out, size, median, mean, mad);

    // Isolate masked pixels as they have high noise
    // n->0 should be smoothed
    std::vector<int> mask_idx;
    for (int i = 0; i < size; ++i)
        if ((n2[i] - median) > 3.5 * mad)
            mask_idx.push_back(i);

    if (use_mean) {
        std::fill_n(out, size, mean);
    } else {
        std::vector<double> tempvector;
        padArray(n2, size, HWSIZE, tempvector);

        if (mask_idx.front() == 0)
            std::fill_n(tempvector.begin(), HWSIZE, mean);

        for (const int &idx : mask_idx)
            tempvector[idx + HWSIZE] = mean;

        if (mask_idx.back() == size - 1)
            std::fill_n(tempvector.end() - HWSIZE, HWSIZE, mean);

        // Convolve
        // std::fill_n(out, size, 0);
        for (int i = 0; i < size; ++i)
            out[i] = cblas_ddot(KS, gaussian_kernel, 1, tempvector.data() + i, 1);
    }

    // Restore original noise for masked pixels
    for (const int &idx : mask_idx)
        out[idx] = n2[idx];
}
