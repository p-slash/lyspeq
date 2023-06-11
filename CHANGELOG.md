# Changelog
+ Poisson bootstrap covariance implemented.
+ Progress counter implemented.
+ `SaveEachChunkResult` saves only the upper triangle of the Fisher matrix.
+ `DynamicChunkNumber` behavior change. `MAX_PIXELS_IN_FOREST = 700` and maximum number of chunks is limited to `DynamicChunkNumber`.
+ `SaveEachChunkResult` option to save each chunk result to FITS file. The fisher matrices are not multiplied by 0.5 and only upper triangle is non-zero.
+ `LookUpTableDir` to save lookup tables instead of being relative to `OutputDir`.
+ Demand all matrices to fit into memory. Skip below Nyquist (not half of nyquist)
+ `dv` of `QSOFile` is not rounded to nearest five, but compared with rounded numbers in sq_table.
+ `Chunk` estimate calculates until Nyquist not half of it.
+ Continuum marginalization now has lambda polynomial templates. New keywords in config are `ContinuumLambdaMargOrder` for lambda polynomials, `ContinuumLogLambdaMargOrder` for log lambda polynomials.
+ `DynamicChunkNumber` to dynamically chunk spectrum into multiple segments. This is achieved by moving quadratic estimator to a new class `Chunk` and using `OneQSOEstimate` as a wrapper for multiple chunks instead. `MAX_PIXELS_IN_FOREST 1000` due to typical performance limitations. If a given spectrum has more pixels than this, resulting `nchunks` will be greater than `DynamicChunkNumber`.
+ Continuum marginalization is now implemented with Sherman-Morrison identity. New option `ContinuumMargOrder` decides the maximum order of `ln lambda`. E.g., `ContinuumMargOrder 1` will marginalize out constant and slope. Old options `ContinuumMargAmp` and `ContinuumMargDerv` are removed.
+ `ResoMatDeconvolutionM (double)` option is added to config file. It deconvolves the resolution matrix with this value if >0. Should be around 1.
+ `SaveEachSpectrumResult` is changed to `SaveEachProcessResult`. This would constrain bootstrap estimation to subsamples determined by the number of processors, but save a lot space and coding.
+ Deconvolution of sinc added while oversampling using `FFTW` package. This deconvolution is needed because resolution matrix is downsampled in 2D.
+ Implemented a 'Smoother' class in QuadraticEstimate. `SmoothNoiseWeights (int)` option is added to config file. If 0, qmle uses the mean noise in covariance matrix as weights. For >0, a Gaussian kernel with sigma equals to this value is applied to the noise. For <0, smoothing is turned off.
+ Pixels in each spectrum is cut below and above the redshift range. Short spectra (Npix < 20) are skipped.
+ Each PE saves its own bootstrap results into one file.
+ Logging only on pe==0. Moved io to std and removed io log.
+ New functionality & config file option `OversampleRmat`. Pass > 0 to oversample resolution in dia matrix form.
+ Removed EdS approximation option. Always use logarithmic velocity conversion.
+ Lookuptables are now saved relative to output directory.
+ Config file has `PrecomputedFisher` option to read file and skip fisher matrix computation.
+ Config file has `InputIsPicca` option to read picca fits files instead. Construct the file list using HDU numbers of each chunk, e.g. third spectrum, `picca-delta-100.fits.gz[3]`.
+ Config file has `UseResoMatrix` option to read resolution matrix from picca file.
+ CFITSIO is a dependency.
+ Config file has `CacheAllSQTables` option to save all sq tables in memory rather than reading one qso at a time.
+ "SaveEachSpectrumResult" in config file saves each spectrum's Fisher and power' estimates
+ R & dv pairs are read from a file, instead of assuming fixed dv for all spectra.
+ Last k edge is now read from config file as `LastKEdge`.
+ Needs CBLAS & LAPACKE libraries to run.
+ Option to turn off copying S and Q matrices in order to reduce time.
+ Logger saves separately for each PE.
+ Use `chrono` library to measure time.
+ GSL flags are silenced in compilation.
+ Tests checking cblas & estimator results are added `make test`.
+ Python script does not fit, but applies a weighted smoothing.
+ Default velocity is logarithmic. Changed the key in config.param
+ Implemented reading a mean flux file, which changed config.param keys
+ Created continuum marginalization coeffecients under specifics namespace. Read from config file.
+ Removed redshift evolution option
+ Each PE reads and sorts its own spectra, then merge sorted. This saves significant amount of time reading ~1m files.
+ Smoothing script has --interp_log option, which can be enabled by setting SmoothLnkLnP to 1 in config.param

### v2
Intermediate versions have been used to study DESI-lite mocks. Further modifications added configuration options, but most importantly optimizated the speed to run on high resolution quasars. This version gave results on full KODIAQ DR2, SQUAD DR1 and XQ-100 samples that are in good agreement with previous measurements.

### mpi-v1.3
+ Signal and derivative look up tables are controlled by one class: [sq_table.hpp](core/sq_table.hpp)
+ Input data are assumed to be fluctuations and not flux. Added `ConvertFromFluxToDeltaf` to config file to switch behaviors.
+ Using Lee12 mean flux option is removed. A better candidate would be interpolating an input flux file. 
+ Signal matrix interpolation axes are reversed to speed up the integration. This makes previous tables uncompetable.
+ MPI branch has been removed of any OpenMP code.
+ Fiducial cosmology has default parameters:

    A      =    6.621420e-02
    n      =   -2.685349e+00
    alpha  =   -2.232763e-01
    B      =    3.591244e+00
    beta   =   -1.768045e-01
    lambda =    3.598261e+02

+ When `FiducialPowerFile` set in config file, an interpolation function takes over. This file should be a binary file and have the following convention:

    Nk Nz
    z[1]...z[Nz]
    k[1]...k[Nk]
    P[nz=1, nk=1...Nk]...P[Nz, nk=1...Nk]

+ Some functions are turned into `inline`.
+ The large number added to covariance matrix is 100.

### v1.2
+ Turn of redshift evolution by passing `--disable-redshift-evolution` to configure script. When redshift evolution is turned off, every pixel pair in each chunk assumed to be at the median redshift of the chunk, and switched to using top-hat bins. However, scaling with a growth function is still functional if passed. Using `beta=0` in fiducial power is recomended.
+ Detailed power spectrum output has build and configuration specifics in the beginning.
+ Log to both `OutputDir` log files and `stdout/stderr`. Save time statistics to `OutputDir/time_log.txt` in fixed width table.
+ Throw `std::runtime_error(msg)` instead of only message.

### v1.1
+ Prints detailed power spectrum output file.
+ Computes Fisher matrix weighted d (raw power), b (noise estimate) and t (fiducial signal estimate) values storing 3 more gsl_vectors.

v1
=====
+ Set `UseLogarithmicVelocity` in config file to 1 to convert by logarithmic spacing.
+ All redshift binning schemes scales derivative matrices with fiducial growth unless `--disable-redshift-growth` is passed to configure or `-DREDSHIFT_GROWTH_POWER` is not passed to preprocessor.
+ Convergence checks if estimated error is negative. Negative errors are substituted with infinite error *only* in convergence---Fisher matrix is not modified.
+ Load balancing takes how many matrices are needed in triangular bins into account. Some bins are skipped since no pixel pair belongs to them.
+ Specify allocated memory in MB to store additional derivative matrices and fiducial signal matrix if possible. This speeds up calculations.
+ The large number added to covariance matrix is 10000.