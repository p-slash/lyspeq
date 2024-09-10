`lyspeq` is highly efficient, parallelized and customizable program for 1D flux power spectrum of the Lyman-alpha forest that implements quadratic maximum likelihood estimator. Please cite following papers:

+ Karaçaylı N.G., Font-Ribera A., Padmanabhan N., 2020, “Optimal 1D Lyα forest power spectrum estimation - I. DESI-lite spectra”, [MNRAS, 497, 4742](https://doi.org/10.1093/mnras/staa2331). [arXiv:2008.06421](https://arxiv.org/abs/2008.06421)
+ Karaçaylı N.G. et al., 2022, “Optimal 1D Lyα forest power spectrum estimation - II. KODIAQ, SQUAD, and XQ-100”, [MNRAS, 509, 2842](https://doi.org/10.1093/mnras/stab3201). [arXiv:2108.10870](https://arxiv.org/abs/2108.10870)
+ Karaçaylı N.G. et al., 2024, “Optimal 1D Lyα Forest Power Spectrum Estimation - III. DESI early data”, [MNRAS, 528, 3941](https://doi.org/10.1093/mnras/stae171). [arXiv:2306.06316](https://arxiv.org/abs/2306.06316)

Prerequisites
=====
+ [GSL](https://www.gnu.org/software/gsl/) is needed for some integration and interpolation.
+ [Python3](https://www.python.org), [Numpy](http://www.numpy.org) and [Scipy](http://www.numpy.org) are needed for fitting.
+ [MPI](https://www.open-mpi.org) is needed to enable parallel computing.
+ CBLAS and LAPACKE. I have been mostly using [MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html#gs.932925) and [OpenBLAS](http://www.openblas.net). [ATLAS](http://math-atlas.sourceforge.net) has been passing simple `make test`, but not fully validated. The compiler flags will depend on the system specifics. Modify Makefile accordingly. To link one of these libraries to `gsl_cblas`, remove `-lgslcblas` from `LDLIBS`. For ATLAS, add `-lcblas -latlas` to `LDLIBS` in your Makefile. To link OpenBLAS, add `-lopenblas`. Intel has [link line advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor).
+ [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) for reading picca files.
+ [FFTW3] (http://fftw.org) for deconvolution of sinc in resolution matrix oversampling.

Compile and Install
=====
The gcc & MKL version can be installed by `./configure x64-linux-gnu-mklxe18 && make && make install`. However, this does not enable MPI. 

MPI can be enabled by passing `--enable-mpi`. 

Run `make test` to make sure everything works ok. Note this does not work with Fisher optimization.

I added some built-in system types. For example, MacBook build is `./configure x64-macos-clang-openblas --enable-mpi`, which will enable OpenMPI and use OpenBLAS for matrix operations. Another important one is Intel compiler with MKL. If you have Parallel Studio XE 2018 installed: `./configure --build=x64-linux-icpc-mklxe18 --enable-mpi`. To see all build types: `./configure --print-builds`.

The executables are installed in `bindir`. This is `/usr/local/bin` by default. You can change it by setting `--bindir=your-path`. Make sure it is in your `$PATH`. Alternatively, you can change `/usr/local` by setting `--prefix` and `--exec_prefix`. Typically for clusters, you want to install binaries to `$HOME/bin`; so simply pass `--prefix=$HOME`.

You can also change interpolation schemes and redshift binning shape. You can also compile in another directory by setting `--srcdir=[source-path]`. For more options run `./configure --help`.

Overview of Programs
=====
+ **LyaPowerEstimate** iterates over multiple qso spectra and estimates one-dimensional Lya power spectrum in the line-of-sight. It maximizes likelihood using Newton-Raphson method. When compiled with MPI compatibility, it distributes qso chunks to multiple CPUs, then reduces the results.
+ **LyaPowerxQmlExposure** is the cross-exposure P1D estimator. The filename input is different than LyaPowerEstimate such that each MPI task reads the entire delta files it was assigned. Each delta file assumed to have all exposures associated with a TARGETID. Do not use more MPI tasks than the number of delta files. The estimator cross correlates exposures with different EXPID and NIGHT, and exposure that have more than 60% overlap in wavelength coverage by default. These can be adjusted in the config file.
+ **CreateSQLookUpTable** creates look up tables for signal and derivative matrices used in LyaPowerEstimate. When compiled with MPI compatibility, it computes tables for different resolution on multiple CPUs.

Both programs take one common config file.

+ **smbivspline.py** is an intermediate smoothing script. An intermediate fitting script is also provided but deprecated.

Config File
=====
See [example file](tests/input/test.config).

Bin edges for k start with linear spacing: `K0 + LinearKBinWidth * n`, where `n=[0, NumberOfLinearBins]`. Then continues with log spacing: `K_Edges[NumberOfLinearBins] * 10^(Log10KBinWidth * n)`. Parameters for k binning are:

    K0 0.

    NumberOfLinearBins    5
    NumberOfLog10Bins     13

    LinearKBinWidth       2e-4
    Log10KBinWidth        0.1

Can add a last bin edge. This goes into effect when larger than the last bin as defined by parameters above.

    LastKEdge 10

Redshift bins are linearly spaced.

    FirstRedshiftBinCenter    1.8
    RedshiftBinWidth          0.2
    NumberOfRedshiftBins      6

Fiducial power can be a tabulated file

    FiducialPowerFile ./my_fiducial_estiamte.dat

Fiducial Palanque fit function parameters when used.
    
    FiducialAmplitude            0.06
    FiducialSlope               -2.6
    FiducialCurvature           -0.1
    FiducialRedshiftPower        0
    FiducialRedshiftCurvature    0
    FiducialLorentzianLambda     0
    
Lookup tables are generated with the following parameters. The resulting velocity spacing will be VelocityLength/(NumberVPoints+1).

    NumberVPoints     200
    NumberZPoints     200
    VelocityLength    1000.

Turn off fiducial signal matrix by setting this to a positive integer integer.
    
    TurnOffBaseline   0

You can smooth lnk, lnP instead of k, P for better behaviour.

    SmoothLnkLnP      1

The maximum number of iterations are (> 1 will yield inaccurate noise bias)

    NumberOfIterations    5

Config file has one file list for qso spectra. This file should start with number of qsos, and then have their relative file paths (see [example file](tests/input/flist.txt)). The location of the file list, and the directory where those files live:

    FileNameList      ./data/qso_dir/qso_list.txt
    FileInputDir      ./data/qso_dir/

The directory for output files and file name base:

    OutputDir         ./data/qso_results/
    OutputFileBase    lya

List of spectograph resolutions and pixel spacings (R [int], dv [double]) is in `FileNameRList`. This file starts with number of lines. See [example file](tests/input/slist.txt).

    FileNameRList     ./data/qso_dir/specres_list.txt

You can save individual results for each process by setting this to 1.

    SaveEachProcessResult 0

These lookup tables are saved with the following file name bases (optional). Using a fixed path for `LookUpTableDir` is recommended:

    LookUpTableDir                 .
    SignalLookUpTableBase          signal_lookup
    DerivativeSLookUpTableBase     derivative_lookup

Specify allocated memory in MB to store additional derivative matrices and fiducial signal matrix if possible.

    AllocatedMemoryMB 5.

Specify the maximum order of `ln lambda` and `lambda` polynomial to marginalize out. E.g., 1 will marginalize out constant and slope. Default is 1. Pass <0 to turn off.

    ContinuumLogLambdaMargOrder  1
    ContinuumLambdaMargOrder     1
    
Specify temporary folder. Smooth power spectrum fitting creates temp files to communicate with [py/lorentzian_fit.py](py/lorentzian_fit.py) script. For clusters, you are typically assigned a scratch space such as `scratch` or `project`.

    TemporaryFolder /tmp
    
It is recommended that flux to fluctuations conversion done using an empirical or theoretical mean flux. Pass this binary filename (int size, double z[size], double flux[size])

    MeanFluxFile ./true_mean_flux.dat

However, if you want to convert using mean flux of each chunk, set the following to 1. This overrides the mean flux file.

    UseChunksMeanFlux  0

If your files have flux fluctuations, set the following to 1. This overrides all above, even nothing was passed.

    InputIsDeltaFlux 0

Other params
====
+ `MinimumSnrCut` (double, default: 0)
    Minimum mean SNR in the forest.
+ `InputIsPicca` (int):
    If > 0, input file format is from picca. Off by default.
+ `SmoothNoiseWeights` (int):
    If > 0, smooths pipeline noise by this Gaussian sigma pixel. Smoothing kernel
    half window size is 25 pixels. If == 0, sets every value to the median.
+ `UseResoMatrix` (int):
    If > 0, reads and uses the resolution matrix picca files.
    Off by default.
+ `SmoothResolutionMatrix` (int, default: -1, False):
    If > 0, uses `SmoothNoiseWeights` value to smooth diagonal resolution matrices.
+ `ResoMatDeconvolutionM` (double):
    Deconvolve the resolution matrix by this factor in terms of pixel.
    For example, 1.0 deconvolves one top hat. Off by default and when
    <= 0.
+ `OversampleRmat` (int):
    Oversample the resolution matrix by this factor per row. Off when <= 0
    and by default.
+ `UseFftMeanResolution` (int):
    Use FFT of the weighted mean resolution in derivative matrix construction
    per chunk.
+ `DynamicChunkNumber` (int):
    Dynamiccaly chunk spectra into this number when > 1. Off by default.
+ `TurnOffBaseline` (int):
    Turns off the fiducial signal matrix if > 0. Fid is on by default.
+ `SmoothLnkLnP` (int):
    Smooth the ln k and ln P values when iterating. On by default
    and when > 0.
+ `ChiSqConvergence` (int):
    Criteria for chi square convergance. Valid when > 0. Default is 1e-2
+ `ContinuumLogLambdaMargOrder` (int):
    Polynomial order for log lambda cont marginalization. Default 1.
+ `ContinuumLambdaMargOrder` (int):
    Polynomial order for lambda cont marginalization. Default -1.
+ `PrecomputedFisher` (str):
    File to precomputed Fisher matrix. If present, Fisher matrix is not
    calculated for spectra. Off by default.
+ `Targetids2Ignore` (str):
    File that contains a list of TARGETIDs. 3D estimator only.
+ `NumberOfBoots` (int, default: 20000):
    Number of bootstrap realizations.
+ `FastBootstrap` (int, default: 1, True):
    Fast bootstrap method. Does not recalculate the Fisher matrix.
+ `SaveBootstrapRealizations` (int, default: 0, False):
    Saves bootstrap realizations.

Cross exposure params
====
+ `DifferentNight` (int, default: 1, True):
    Cross correlate different nights only.
+ `DifferentFiber` (int, default: -1, False):
    Cross correlate different fibers only.
+ `DifferentPetal` (int, default: -1, False):
    Cross correlate different petals only.
+ `MinXWaveOverlapRatio` (double, default: 0.6):
    Cross correlate segments that overlap more than this ratio.

Quasar Spectrum File
====
## Binary format
It starts with a header (see [QSOFile](io/qso_file.hpp)), then has wavelength, fluctuations and noise in double arrays. A Python script is added to help conversion between different formats (see [BinaryQSO](py/binary_qso.py)). When using this format, end files with `.dat` or `.bin` extensions.

## Picca format
**LyaPowerEstimate format**: When using this format, construct the file list using HDU numbers of each chunk. E.g., for the third spectrum, use picca-delta-100.fits.gz[3]. This is what filename list should look like:

    3
    picca-delta-100.fits.gz[1]
    picca-delta-100.fits.gz[2]
    picca-delta-100.fits.gz[3]

**LyaPowerxQmlExposure format**: When using this format, pass the entire delta file. Do not use more MPI tasks than number of files.

    3
    picca-delta-0.fits
    picca-delta-1.fits
    picca-delta-2.fits

**Following keys are read from the header:**

+ Number of pixels is `NAXIS2`.
+ ID is `TARGETID (long)`.
+ Redshift of the quasar is `Z (double)`.
+ **LyaPowerxQmlExposure format** needs `EXPID (int)` and `NIGHT (int)`.
+ `MEANRESO (double)` is assumed to be the Gaussian R value in km/s. This is converted to integer FWHM resolving power by rounding up last two digits.

        fwhm_resolution = int(SPEED_OF_LIGHT / MEANRESO / ONE_SIGMA_2_FWHM / 100 + 0.5) * 100;

+ `MEANSNR (double)` is read and used to drop low SNR forests based on MinimumSnrCut key in the config file.
+ Pixel spacing is read from `DLL (double)` (difference of log10 lambda), then converted to km/s units.

        dv_kms = DLL * SPEED_OF_LIGHT * ln(10);

**Following data are read from the data tables:**

+ `LAMBDA` as wavelength. Or `LOGLAM` as log10(lambda), which is converted back to lambda.
+ Flux fluctuations are read from `DELTA`.
+ Inverse variance is read from `IVAR`. This is converted back to sigma.
+ When the option is set, the resolution matrix is read from `RESOMAT`. This is reordered for C arrays.

Bootstrap file output
===
When `SaveEachProcessResult 1` is passed in the config file, individual results from each process will be saved into files `OutputDir`. All results will be saved to `bootresults.dat`. This file is in binary format. It starts with two integers for `Nk, Nz` and another integer for `ndiags`. Each result is then a double array of size `cf_size+N` in which `Pk` is the first `N=Nk*Nz` element, and `CompressedFisherMatrix` is in the remaining part. Only the upper diagonals (starting with the main) of the Fisher matrix is saved in order to save space. Note this `Pk` value is before multiplication by fisher inverse. `cf_size = TOTAL_KZ_BINS*ndiags - (ndiags*(ndiags-1))/2` and `ndiags=3*NUMBER_OF_K_BANDS`. This Fisher matrix is multiplied by 0.5.

When `SaveEachChunkResult 1` is passed in the config file, individual results from each **chunk** will be saved to FITS files. Each process will have its own file. Each chunk is written to image extensions. All dk, nk, tk and fisher matrix is saved for only valid bins for each chunk. This is given by 'ISTART' key in header. 'NQDIM' key gives the dimension. Fisher matrix is **not** multiplied by 0.5 and only the upper triangle is saved. Sample code to add chunk fisher to total fisher after converting to NQDIM x NQDIM:

```
for (int i_kz = 0; i_kz < N_Q_MATRICES; ++i_kz) {
    int idx_fji_0 =
        (TOTAL_KZ_BINS + 1) * (i_kz + ISTART);
    int ncopy = NQDIM - i_kz;
    mxhelp::vector_add(
        fisher_total + idx_fji_0,
        fisher_chunk.get() + i_kz * (NQDIM + 1),
        ncopy);
}
```

Poisson Bootstrapping
===
`lyspeq` performs a parallel Poisson bootstrapping in the end. Poisson bootstrapping generates random coefficients for each quasar (not chunk) using `Poisson(mu=1)` random distribution. This approximation is based on Binomial distribution for large n. The sum of these coefficients are not constrained to be the total number of quasars in the sample, which could be refined at future versions if necesary [1](https://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html), [2](http://www.med.mcgill.ca/epidemiology/Hanley/Reprints/bootstrap-hanley-macgibbon2006.pdf), [3](https://mihagazvoda.com/posts/poisson-bootstrap/).

### Highlighted Features
+ Poisson bootstrap covariance estimate.
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

+ Intermediate python script applies a weighted smoothing. This smoothing script has --interp_log option, which can be enabled by setting `SmoothLnkLnP` to 1 in the config file.
+ `SaveEachChunkResult` in config file saves each chunk's Fisher and power estimates.

### Optimizations
+ Each PE reads and sorts its own spectra, then they all perform a merge sort. This saves significant amount of time reading ~1m files.
+ Using discrete interpolation instead of `gsl_interp` to eliminate time spent on binary search.
+ Does not copy S and Q matrices when they are not changed, which saves time.
+ Use `chrono` library to measure time.
+ No more `gsl_vector` and `gsl_matrix`, but uses LAPACKE instead.

### Other
+ Logger saves separately for each PE.
+ Needs CBLAS & LAPACKE libraries to run (not using `gsl_vector` and `gsl_matrix`).
+ GSL flags are silenced in compilation.
+ Removed redshift evolution option.







