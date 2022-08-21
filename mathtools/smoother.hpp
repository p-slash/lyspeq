#ifndef SMOOTHER_H
#define SMOOTHER_H

class Smoother
{
    // SmoothNoiseWeights = 0 is mean noise
    // < 0 uses raw noise
    // > 0 sets the sigma pixels
    #define HWSIZE 25
    #define KS 2*HWSIZE+1

    static int sigmapix;
    static double gaussian_kernel[KS];
    static bool isKernelSet, useMedianNoise, isSmoothingOn;

public:
    static void setParameters(int noisefactor);
    static void setGaussianKernel();
    static void smoothNoise(const double *n2, double *out, int size);

    #undef HWSIZE
    #undef KS
};

#endif
