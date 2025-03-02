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
#define THRESHOLD_VAR 1E6


void padArray(
        const double *arr, int size, int hwsize, std::vector<double> &out
) {
    out.resize(size + 2 * hwsize, 0);

    std::copy_n(arr, size, out.begin() + hwsize);
    std::fill_n(out.begin(), hwsize, arr[0]);
    std::fill_n(out.begin() + size, hwsize, arr[size - 1]);
}


double _getMedianBelowThreshold(
        double *sorted_arr, int size, int &newsize
) {
    newsize = size;
    while ((newsize != 0) && sorted_arr[newsize - 1] > THRESHOLD_VAR)
        --newsize;

    if (newsize == 0)
        return sorted_arr[0];

    return stats::medianOfSortedArray(sorted_arr, newsize);
}


void _findMedianStatistics(
        double *arr, int size, double &median, double &mean, double &mad
) {
    int newsize;
    std::sort(arr, arr + size);
    median = _getMedianBelowThreshold(arr, size, newsize);
    mean = std::accumulate(arr, arr + newsize, 0.) / newsize;

    std::for_each(arr, arr + newsize, [median](double &f) { f = fabs(f - median); });
    std::sort(arr, arr + newsize);

    // The constant factor makes it unbiased
    mad = 1.4826 * stats::medianOfSortedArray(arr, newsize);
}


double _findInterpolatingFinite(const double *in, int j, int size) {
    double l = 0, u = 0;
    int m, n;
    for (m = 1; j - m >= 0; ++m) {
        if ((in[j - m] < THRESHOLD_VAR)) {
            l = in[j - m];
            break;
        }
    }

    for (n = 1; j + n < size; ++n) {
        if (in[j + n] < THRESHOLD_VAR) {
            u = in[j + n];
            break;
        }
    }

    if (l == 0)  return u;
    if (u == 0)  return l;

    return (l * n + u * m) / (m + n);
}


void _fillMaskedRegion(
        const std::vector<int> &mask_idx, int HWSIZE,
        int size, std::vector<double> &v
) {
    if (mask_idx.empty())
        return;

    double *in = v.data() + HWSIZE;
    for (const int &idx : mask_idx) {
        if (in[idx] < THRESHOLD_VAR)
            v[idx + HWSIZE] = in[idx];
        else
            v[idx + HWSIZE] = _findInterpolatingFinite(in, idx, size);
    }

    std::fill_n(v.begin(), HWSIZE, v[HWSIZE]);
    std::fill_n(v.end() - HWSIZE, HWSIZE, v[size - 1 + HWSIZE]);
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


void Smoother::smoothIvar(const double *ivar, double *out, int size) {
    DEBUG_LOG("Smoothing ivar.\n");
    double count = 0, mean = 0;

    // measure old statistics "n2"
    std::transform(
        ivar, ivar + size, out, [](const double &iv) {
            if (iv == 0)
                return 1e15;
            return 1.0 / iv;
        });

    for (int i = 0; i < size; ++i) {
        if (out[i] > THRESHOLD_VAR)  continue;
        mean += out[i];
        count += 1.0;
    }
    mean /= count;

    std::vector<double> medfilt_var = stats::medianFilter(out, size, KS);
    for (int i = 0; i < size; ++i)
        out[i] = fabs(out[i] - medfilt_var[i]) + 1e-8 + out[i] * 1e-6;

    std::vector<double> mad = stats::medianFilter<stats::REFLECT>(out, size, KS);
    for (int i = 0; i < size; ++i)
        mad[i] = 1.0 / std::min(THRESHOLD_VAR, medfilt_var[i] + 5 * 1.4826 * mad[i]);

    // Isolate masked pixels as they have high noise
    // n->0 should be smoothed
    std::vector<int> mask_idx;
    for (int i = 0; i < size; ++i)
        if (ivar[i] < mad[i])
            mask_idx.push_back(i);

    if (use_mean) {
        std::fill_n(out, size, mean);
    } else {
        // Smooth variance to preserve total variance as much as possible.
        std::vector<double> tempvector(size + 2 * HWSIZE);

        std::transform(
            ivar, ivar + size, tempvector.begin() + HWSIZE,
            [](const double &iv) {
                if (iv == 0)
                    return 1e15;
                return 1.0 / iv;
            }
        );

        for (const int &idx : mask_idx)
            tempvector[idx + HWSIZE] = 1e15;

        _fillMaskedRegion(mask_idx, HWSIZE, size, tempvector);

        // Convolve
        // std::fill_n(out, size, 0);
        for (int i = 0; i < size; ++i)
            out[i] = 1.0 / cblas_ddot(KS, gaussian_kernel, 1, tempvector.data() + i, 1);
    }

    // Restore original noise for masked pixels
    for (const int &idx : mask_idx)
        out[idx] = ivar[idx];
}
