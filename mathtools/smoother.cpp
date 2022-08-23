#include "mathtools/smoother.hpp"
#include <cmath>
#include <algorithm> // std::for_each
#include <vector>

#ifdef USE_MKL_CBLAS
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

void _findMedianStatistics(double *arr, int size, double &median, double &mad)
{
    std::sort(arr, arr+size);
    median = arr[size/2];

    std::for_each(arr, arr+size, [median](double &f) { f = fabs(f-median); });
    std::sort(arr, arr+size);
    mad = 1.4826 * arr[size/2]; // The constant factor makes it unbiased
}

#define HWSIZE 25
#define KS 2*HWSIZE+1

int Smoother::sigmapix;
double Smoother::gaussian_kernel[KS];
bool Smoother::isKernelSet = false, Smoother::useMedianNoise = false, Smoother::isSmoothingOn = false;

void Smoother::setParameters(int noisefactor)
{
    // NOISE_SMOOTHING_FACTOR = 0 is mean noise
    // < 0 uses raw noise
    // > 0 sets the sigma pixels
    if (noisefactor >= 0) 
        Smoother::isSmoothingOn = true;
    if (noisefactor == 0)
        Smoother::useMedianNoise = true;

    Smoother::sigmapix = noisefactor;
}

void Smoother::setGaussianKernel()
{
    if (Smoother::isKernelSet || !Smoother::isSmoothingOn || Smoother::useMedianNoise)
        return;

    double sum=0, *g = &Smoother::gaussian_kernel[0];
    for (int i = -HWSIZE; i < HWSIZE+1; ++i, ++g)
    {
        *g = exp(-pow(i*1./Smoother::sigmapix, 2)/2);
        sum += *g;
    }

    std::for_each(&Smoother::gaussian_kernel[0], &Smoother::gaussian_kernel[0]+KS, 
        [&](double &x) { x /= sum; });

    Smoother::isKernelSet = true;
}

void Smoother::smoothNoise(const double *n2, double *out, int size)
{
    if (!Smoother::isSmoothingOn)
    {
        std::copy(&n2[0], &n2[0] + size, &out[0]); 
        return;
    }

    double median, mad;

    // convert square of noise to noise
    std::transform(n2, n2+size, out, [](double x2) { return sqrt(x2); });
    _findMedianStatistics(out, size, median, mad);

    if (Smoother::useMedianNoise)
    {
        std::fill_n(out, size, median*median);
        return;
    }

    // Isolate masked pixels as they have high noise
    std::vector<int> mask_idx;
    std::vector<double> padded_noise(size+2*HWSIZE, 0);

    for (int i = 0; i < size; ++i)
    {
        double n = sqrt(n2[i]);
        // n->0 should be smoothed
        if ((n-median) > 3.5*mad)
        {
            mask_idx.push_back(i);
            padded_noise[i+HWSIZE] = median;
        }
        else
            padded_noise[i+HWSIZE] = n;
    }

    // Replace their values with median noise
    // for (auto it = mask_idx.begin(); it != mask_idx.end(); ++it)
    //     padded_noise[*it+HWSIZE] = median;
    // Pad array by the edge values
    for (int i = 0; i < HWSIZE; ++i)
    {
        padded_noise[i] = padded_noise[HWSIZE];
        padded_noise[i+HWSIZE+size] = padded_noise[HWSIZE+size-1];
    }

    // Convolve
    std::fill_n(out, size, 0);
    for (int i = 0; i < size; ++i)
    {
        out[i] = cblas_ddot(KS, Smoother::gaussian_kernel, 1, padded_noise.data()+i, 1);
        out[i] *= out[i];
    }

    // Restore original noise for masked pixels
    for (auto it = mask_idx.begin(); it != mask_idx.end(); ++it)
        out[*it] = n2[*it];
}

#undef HWSIZE
#undef KS
