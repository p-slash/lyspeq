`lyspeq` is highly efficient, parallelized and customizable program for 1D flux power spectrum of the Lyman-alpha forest that implements quadratic maximum likelihood estimator. Please cite papers Karaçaylı et al. (2020) and Karaçaylı et al. (submitted to MNRAS).

+ Karaçaylı N. G., Font-Ribera A., Padmanabhan N., 2020, [MNRAS](https://doi.org/10.1093/mnras/staa2331), 497, 4742
+ Karaçaylı N. G., et al., 2021, MNRAS, submitted

# Changelog
+ Config file has 'PrecomputedFisher' option to read file and skip fisher matrix computation.
+ Config file has 'InputIsPicca' option to read picca fits files instead. Construct the file list using HDU numbers of each chunk, e.g. third spectrum, picca-delta-100.fits.gz[3]. WARNING: This option cannot yet save individual results for bootstrapping.
+ Config file has 'UseResoMatrix' option to read resolution matrix from picca file.
+ CFITSIO is a dependency.
+ Config file has 'CacheAllSQTables' option to save all sq tables in memory rather than reading one qso at a time.

# v2
Intermediate versions have been used to study DESI-lite mocks. Further modifications added configuration options, but most importantly optimizated the speed to run on high resolution quasars. This version 2.0 already gave preliminary results on full KODIAQ DR2 and XQ-100 sample that are in good agreement with previous measurements.

### Highlighted Features
+ Input data are assumed to be fluctuations and not flux. Added `ConvertFromFluxToDeltaf` to config file to switch behaviors.
+ Default velocity is logarithmic. Change this by setting `UseEDSVelocity` to 1 in config file.
+ R & dv pairs are read from a file, instead of assuming fixed dv for all spectra.
+ A detailed python3 script in `make test` for CBLAS, SQ tables and the estimator results.
+ Specify allocated memory in MB to store additional derivative matrices and fiducial signal matrix if possible. This speeds up calculations.
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
+ "SaveEachSpectrumResult" in config file saves each spectrum's Fisher and power' estimates.
+ Last k edge is now read from config file as `LastKEdge`.
+ Implemented reading a mean flux file, which can be read by `MeanFluxFile` from the config file.

### Optimizations
+ Each PE reads and sorts its own spectra, then merge sorted. This saves significant amount of time reading ~1m files.
+ Using discrete interpolation instead of `gsl_interp` to eliminate time spent on binary search.
+ Does not copy S and Q matrices when they are not changed which save time.
+ Use `chrono` library to measure time.
+ No more `gsl_vector` and `gsl_matrix`, but use LAPACKE instead.

### Other
+ Logger saves separately for each PE.
+ Needs CBLAS & LAPACKE libraries to run, because not using `gsl_vector` and `gsl_matrix`.
+ GSL flags are silenced in compilation.
+ Removed redshift evolution option.

Prerequisites
=====
+ [GSL](https://www.gnu.org/software/gsl/) is needed for some integration and interpolation.
+ [Python3](https://www.python.org), [Numpy](http://www.numpy.org) and [Scipy](http://www.numpy.org) are needed for fitting.
+ [MPI](https://www.open-mpi.org) is needed to enable parallel computing.
+ [CBLAS] and [LAPACKE].
+ [CFITSIO]

Even though [GSL](https://www.gnu.org/software/gsl/) has built in CBLAS functions, I recommended using an optimized library such as Intel's [MKL](https://software.intel.com/en-us/mkl), [ATLAS](http://math-atlas.sourceforge.net) or [OpenBLAS](https://www.openblas.net). The compiler flags will depend on the system specifics. Modify Makefile accordingly. To link one of these libraries to gsl_cblas, remove `-lgslcblas` from `LDLIBS`. For ATLAS, add `-lcblas -latlas` to `LDLIBS` in your Makefile. To link OpenBLAS, add `-lopenblas`. Intel has [link line advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor).

Compile and Install
=====
The gcc & MKL version can be installed by `./configure x64-linux-gnu-mklxe18 && make && make install`. However, this does not enable MPI. 

MPI can be enabled by passing `--enable-mpi`. 

Run `make test` to make sure everything works ok.

I added some built-in system types. For example, MacBook build is `./configure x64-macos-clang-openblas --enable-mpi`, which will enable OpenMPI and use OpenBLAS for matrix operations. Another important one is Intel compiler with MKL. If you have Parallel Studio XE 2018 installed: `./configure --build=x64-linux-icpc-mklxe18 --enable-mpi`. To see all build types: `./configure --print-builds`.

The executables are installed in `bindir`. This is `/usr/local/bin` by default. You can change it by setting `--bindir=your-path`. Make sure it is in your `$PATH`. Alternatively, you can change `/usr/local` by setting `--prefix` and `--exec_prefix`. Typically for clusters, you want to install binaries to `$HOME/bin`; so simply pass `--prefix=$HOME`.

You can also change interpolation schemes and redshift binning shape. You can also compile in another directory by setting `--srcdir=[source-path]`. For more options run `./configure --help`.

Overview of Programs
=====
+ **LyaPowerEstimate** iterates over multiple qso spectra and estimates one-dimensional Lya power spectrum in the line-of-sight. It maximizes likelihood using Newton-Raphson method. When compiled with MPI compatibility, it distributes qso chunks to multiple CPUs, then reduces the results.
+ **CreateSQLookUpTable** creates look up tables for signal and derivative matrices used in LyaPowerEstimate. When compiled with MPI compatibility, it computes tables for different resolution on multiple CPUs.

Both programs take one common config file.

+ **smbivspline.py** is an intermediate smoothing script. An intermediate fitting script is also provided by deprecated.

Config File
=====
Bin edges for k start with linear spacing: `K0 + LinearKBinWidth * n`, where `n=[0, NumberOfLinearBins]`. Then continues with log spacing: `K_Edges[NumberOfLinearBins] * 10^(Log10KBinWidth * n)`. Parameters for k binning are:

    K0 0.

    NumberOfLinearBins    5
    NumberOfLog10Bins     13

    LinearKBinWidth       2E-4
    Log10KBinWidth        0.1

Can add a last bin edge. This goes into effect when larger than the last bin as defined by parameters above.

    LastKEdge 10

Redshift bins are linearly spaced.

    FirstRedshiftBinCenter    1.9
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
    
Lookup tables are generated with the following parameters:

    NumberVPoints     200
    NumberZPoints     200
    VelocityLength    1000.

Turn off fiducial signal matrix by setting this to a positive integer integer.
    
    TurnOffBaseline   0

You can smooth lnk, lnP instead of k, P for better behaviour.

    SmoothLnkLnP      1

The maximum number of iterations are

    NumberOfIterations    10

Config file has one file list for qso spectra. This file should start with number of qsos, and then have their relative file paths. The location of the file list, and the directory where those files live:

    FileNameList      ./data/qso_dir/qso_list.txt
    FileInputDir      ./data/qso_dir/

The directory for output files and file name base:

    OutputDir         ./data/qso_results/
    OutputFileBase    lya

List of spectograph resolutions and pixel spacings (R [int], dv [double]) is in `FileNameRList`. This file starts with number of lines.

    FileNameRList     ./data/qso_dir/specres_list.txt

You can save individual results for each spectra by setting this to 1.

    SaveEachSpectrumResult 0

These lookup tables are saved with the follwoing file name bases to `FileInputDir`:

    SignalLookUpTableBase          signal_lookup
    DerivativeSLookUpTableBase     derivative_lookup

Specify allocated memory in MB to store additional derivative matrices and fiducial signal matrix if possible.

    AllocatedMemoryMB 5.

Specify continuum marginalization coefficients. Default is 100. Pass <=0 to turn off.

    ContinuumMargAmp  100.
    ContinuumMargDerv 100.
    
Specify temporary folder. Smooth power spectrum fitting creates temp files to communicate with [py/lorentzian_fit.py](py/lorentzian_fit.py) script. For clusters, you are typically assigned a scratch space such as `scratch` or `project`.

    TemporaryFolder /tmp

To convert with EdS velocity, set this to 1. Default 0 will use logarithmic spacing approximation.

    UseEDSVelocity 0
    
It is recommended that flux to fluctuations conversion done using an empirical or theoretical mean flux. Pass this binary filename (int size, double z[size], double flux[size])

    MeanFluxFile ./true_mean_flux.dat

However, if you want to convert using mean flux of each chunk, set the following to 1. This overrides the mean flux file.

    UseChunksMeanFlux  0

If your files have flux fluctuations, set the following to 1. This overrides all above, even nothing was passed.

    InputIsDeltaFlux 0

Quasar Spectrum File
====
Quasar spectrum file is in binary format. It starts with a header (see [QSOFile](io/qso_file.hpp)), then has wavelength, fluctuations and noise in double arrays. A Python script is added to help conversion between different formats (see [BinaryQSO](py/binary_qso.py)).