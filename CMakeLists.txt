cmake_minimum_required(VERSION 3.10)

project(lyspeq)

option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_MPI "Enable MPI" ON)
option(USE_OPENBLAS_BREW "Use OpenBlas brew" OFF)
option(USE_MKL_LIB "Use MKL library" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(COMMAND_EXE ./)

find_package(PkgConfig REQUIRED)

if (USE_OPENBLAS_BREW)
    # set(ENV{CMAKE_PREFIX_PATH} "/opt/homebrew/opt/openblas")
    # set(BLA_VENDOR OpenBLAS)
    set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/opt/openblas/lib/pkgconfig")
    pkg_check_modules(BLAS openblas REQUIRED)
elseif (USE_MKL_LIB)
    add_definitions(-DUSE_MKL_CBLAS)
    pkg_check_modules(BLAS mkl REQUIRED)
else()
    find_package(BLAS REQUIRED)
endif()

if (ENABLE_OPENMP)
    find_package(OPENMP REQUIRED)
    add_definitions(-DENABLE_OMP)
endif()

if (ENABLE_MPI)
    add_definitions(-DENABLE_MPI)
    pkg_search_module(MPI mvapich2 mpich mpi ompi)
    set(COMMAND_EXE "mpirun -np 1 ")
    # set(MPI_SKIP_COMPILER_WRAPPER ON)
    # find_package(MPI REQUIRED)
    # message(${MPI_VERSION})
    # message("com opt" ${MPI_COMPILE_OPTIONS})
    # message("include dir" ${MPI_INCLUDE_DIRS})
    # message("lib" ${MPI_LIBRARIES})
    # message("flags" ${MPI_CFLAGS})
    # set(MPI_LINKS_MINE mpi)
endif()

pkg_check_modules(CFITSIO cfitsio REQUIRED)
# message(${CFITSIO_LIBRARY_DIRS})
pkg_check_modules(GSL gsl REQUIRED)
pkg_check_modules(FFTW3 fftw3 REQUIRED)

list(APPEND SOURCES
    core/global_numbers.cpp
    core/fiducial_cosmology.cpp
    mathtools/matrix_helper.cpp
    core/one_qso_estimate.cpp
    core/quadratic_estimate.cpp
    core/sq_table.cpp
    mathtools/discrete_interpolation.cpp
    mathtools/fourier_integrator.cpp
    mathtools/interpolation.cpp
    mathtools/interpolation_2d.cpp
    io/config_file.cpp
    io/io_helper_functions.cpp
    io/qso_file.cpp
    io/sq_lookup_table_file.cpp
    io/logger.cpp
    mathtools/real_field.cpp
    core/chunk_estimate.cpp
    mathtools/smoother.cpp
    io/bootstrap_file.cpp
    mathtools/stats.cpp
)

list(APPEND GLOBAL_SOURCES
    core/global_numbers.cpp io/config_file.cpp io/logger.cpp
    io/io_helper_functions.cpp
)

list(APPEND SQ_TABLE_SOURCES
    core/sq_table.cpp io/sq_lookup_table_file.cpp
    core/fiducial_cosmology.cpp mathtools/fourier_integrator.cpp
    mathtools/discrete_interpolation.cpp mathtools/interpolation.cpp
    mathtools/interpolation_2d.cpp
)

include_directories(.)

add_compile_options(
    ${MPI_CFLAGS} ${GSL_CFLAGS}
    ${BLAS_CFLAGS} ${CFITSIO_CFLAGS} ${FFTW3_CFLAGS}
)

link_directories(
    ${MPI_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS} ${BLAS_LIBRARY_DIRS}
    ${CFITSIO_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS}
)
link_libraries(
    stdc++ gsl ${BLAS_LIBRARIES} ${CFITSIO_LIBRARIES} ${FFTW3_LIBRARIES} m
    ${MPI_LIBRARIES}
)

add_executable(LyaPowerEstimate
    LyaPowerEstimate.cpp ${SOURCES})
add_executable(LyaPowerxQmlExposure
    LyaPowerxQmlExposure.cpp ${SOURCES}
    cross/cross_exposure.cpp cross/one_qso_exposures.cpp)
add_executable(CreateSQLookUpTable
    CreateSQLookUpTable.cpp ${GLOBAL_SOURCES} ${SQ_TABLE_SOURCES})

# Test exes
add_executable(cblas_tests
    tests/cblas_tests.cpp tests/test_utils.cpp
    mathtools/matrix_helper.cpp mathtools/real_field.cpp
    mathtools/discrete_interpolation.cpp)
add_executable(testSQCMatrices
    tests/testSQCMatrices.cpp tests/test_utils.cpp ${SOURCES})

include(CTest)
add_test(NAME TestCBLAS COMMAND cblas_tests)
add_test(NAME TestSqMatrices COMMAND testSQCMatrices tests/input/test.config)
file(MAKE_DIRECTORY tests/output)
add_test(NAME TestQmle COMMAND LyaPowerEstimate tests/input/test.config)

