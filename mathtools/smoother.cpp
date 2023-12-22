#include "mathtools/smoother.hpp"
#include "io/logger.hpp"

#include <cmath>
#include <algorithm> // std::for_each
#include <vector>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

std::unique_ptr<Smoother> process::noise_smoother;


double _medianOfSortedArray(double *sorted_arr, int size) {
    int jj = size / 2;
    double median = sorted_arr[jj];

    if (size % 2 == 0) {
        median += sorted_arr[jj - 1];
        median /= 2;
    }

    return median;
}


double _getMedianBelowThreshold(
        double *sorted_arr, int size, int &newsize, double thres=1e3
) {
    newsize = size;
    while ((newsize != 0) && sorted_arr[newsize - 1] > thres)
        --newsize;

    if (newsize == 0)
        return sorted_arr[0];

    return _medianOfSortedArray(sorted_arr, newsize);
}


void _findMedianStatistics(
        double *arr, int size, double &median, double &mad, double thres=1e3
) {
    int newsize;
    std::sort(arr, arr + size);
    median = _getMedianBelowThreshold(arr, size, newsize, thres);

    std::for_each(arr, arr + newsize, [median](double &f) { f = fabs(f - median); });
    std::sort(arr, arr + newsize);

    // The constant factor makes it unbiased
    mad = 1.4826 * _medianOfSortedArray(arr, newsize);
}


Smoother::Smoother(ConfigFile &config)
{
    LOG::LOGGER.STD("###############################################\n");
    LOG::LOGGER.STD("Reading noise smoothing parameters from config.\n");

    config.addDefaults(smoother_default_parameters);
    sigmapix = config.getInteger("SmoothNoiseWeights");
    LOG::LOGGER.STD("SmoothNoiseWeights is set to %d.\n\n", sigmapix);
 
    isSmoothingOn  = (sigmapix >= 0);
    useMedianNoise = (sigmapix == 0);

    if (!isSmoothingOn || useMedianNoise)
        return;

    double sum=0, *g = &gaussian_kernel[0];
    for (int i = -HWSIZE; i < HWSIZE+1; ++i, ++g)
    {
        *g = exp(-pow(i*1./sigmapix, 2)/2);
        sum += *g;
    }

    std::for_each(&gaussian_kernel[0], &gaussian_kernel[0]+KS, 
        [sum](double &x) { x /= sum; });
}

void Smoother::smoothNoise(const double *n2, double *out, int size)
{
    if (!isSmoothingOn)
    {
        std::copy(&n2[0], &n2[0] + size, &out[0]); 
        return;
    }

    double median, mad;

    // convert square of noise to noise
    std::transform(n2, n2 + size, out, [](double x2) { return sqrt(x2); });
    _findMedianStatistics(out, size, median, mad);

    if (useMedianNoise)
    {
        std::fill_n(out, size, median*median);
        return;
    }

    // Isolate masked pixels as they have high noise
    std::vector<int> mask_idx;
    std::vector<double> padded_noise(size + 2 * HWSIZE, 0);

    for (int i = 0; i < size; ++i)
    {
        double n = sqrt(n2[i]);
        // n->0 should be smoothed
        if ((n - median) > 3.5 * mad)
        {
            mask_idx.push_back(i);
            padded_noise[i + HWSIZE] = median;
        }
        else
            padded_noise[i + HWSIZE] = n;
    }

    // Replace their values with median noise
    // for (auto it = mask_idx.begin(); it != mask_idx.end(); ++it)
    //     padded_noise[*it+HWSIZE] = median;
    // Pad array by the edge values
    for (int i = 0; i < HWSIZE; ++i)
        padded_noise[i] = padded_noise[HWSIZE];
    for (int i = 0; i < HWSIZE; ++i)
        padded_noise[i + HWSIZE + size] = padded_noise[HWSIZE + size - 1];

    // Convolve
    // std::fill_n(out, size, 0);
    for (int i = 0; i < size; ++i)
    {
        out[i] = cblas_ddot(KS, gaussian_kernel, 1, padded_noise.data() + i, 1);
        out[i] *= out[i];
    }

    // Restore original noise for masked pixels
    for (const auto &idx : mask_idx)
        out[idx] = n2[idx];
}

