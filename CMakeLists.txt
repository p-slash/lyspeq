cmake_minimum_required(VERSION 3.10)

project(lyspeq)

option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_MPI "Enable MPI" ON)
option(USE_OPENBLAS_BREW "Use OpenBlas brew" OFF)
option(USE_MKL_LIB "Use MKL library" OFF)
option(DEBUG "Compile with debugging options" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(COMMAND_EXE ./)
# message(${CMAKE_SYSTEM})
# if (APPLE)
#     message(aplle)
# endif()

if (DEBUG)
    add_compile_options(-g -std=c++20)
    add_definitions(-DDEBUG)
else()
    add_compile_options(-O3 -std=c++20)
endif()

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
    pkg_search_module(BLAS mkl openblas atlas libsci REQUIRED)
endif()

if (BLAS_MODULE_NAME STREQUAL "mkl")
    add_definitions(-DUSE_MKL_CBLAS)
endif()

if (ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DENABLE_OMP)
endif()

if (ENABLE_MPI)
    add_definitions(-DENABLE_MPI)
    pkg_search_module(MPI mvapich2 mpich mpi ompi REQUIRED)
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

pkg_check_modules(GSL gsl REQUIRED)
# GSL inline definition
add_definitions(-DHAVE_INLINE)

pkg_check_modules(FFTW3 fftw3 REQUIRED)

add_compile_options(
    ${OpenMP_CXX_FLAGS} ${MPI_CFLAGS} ${GSL_CFLAGS}
    ${BLAS_CFLAGS} ${CFITSIO_CFLAGS} ${FFTW3_CFLAGS}
)

link_directories(
    ${MPI_LIBRARY_DIRS} ${GSL_LIBRARY_DIRS} ${BLAS_LIBRARY_DIRS}
    ${CFITSIO_LIBRARY_DIRS} ${FFTW3_LIBRARY_DIRS}
)
link_libraries(
    ${OpenMP_CXX_LIB_NAMES}  ${MPI_LIBRARIES} gsl ${BLAS_LIBRARIES} ${CFITSIO_LIBRARIES}
    ${FFTW3_LIBRARIES} m
)

include_directories(.)

add_library(CoreObjLibs OBJECT
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

add_library(GlobalObjLibs OBJECT
    core/global_numbers.cpp io/config_file.cpp io/logger.cpp
    io/io_helper_functions.cpp
)

add_library(SqTableObjLibs OBJECT
    core/sq_table.cpp io/sq_lookup_table_file.cpp
    core/fiducial_cosmology.cpp mathtools/fourier_integrator.cpp
    mathtools/discrete_interpolation.cpp mathtools/interpolation.cpp
    mathtools/interpolation_2d.cpp
)

add_custom_target(print_genex
    COMMAND ${CMAKE_COMMAND} -E echo 'Objects $<FILTER:$<TARGET_OBJECTS:CoreObjLibs>,EXCLUDE,/quadratic_estimate.cpp.o>'
)

add_executable(LyaPowerEstimate
    LyaPowerEstimate.cpp $<TARGET_OBJECTS:CoreObjLibs>)
add_executable(LyaPowerxQmlExposure
    LyaPowerxQmlExposure.cpp $<TARGET_OBJECTS:CoreObjLibs>
    cross/cross_exposure.cpp cross/one_qso_exposures.cpp)
add_executable(CreateSQLookUpTable
    CreateSQLookUpTable.cpp $<TARGET_OBJECTS:GlobalObjLibs>
    $<TARGET_OBJECTS:SqTableObjLibs>)
add_executable(lyspeqBootStats
    lyspeqBootStats.cpp
    $<FILTER:$<TARGET_OBJECTS:CoreObjLibs>,EXCLUDE,/quadratic_estimate.cpp.o>)

install(TARGETS LyaPowerEstimate LyaPowerxQmlExposure CreateSQLookUpTable lyspeqBootStats
        RUNTIME DESTINATION bin)
install(PROGRAMS py/smbivspline.py DESTINATION bin)

# Test exes
add_executable(cblas_tests
    tests/cblas_tests.cpp tests/test_utils.cpp
    mathtools/matrix_helper.cpp mathtools/real_field.cpp
    mathtools/discrete_interpolation.cpp
)
target_compile_definitions(cblas_tests PRIVATE SRCDIR="${CMAKE_SOURCE_DIR}")
add_executable(testSQCMatrices
    tests/testSQCMatrices.cpp tests/test_utils.cpp $<TARGET_OBJECTS:CoreObjLibs>)

enable_testing()
file(MAKE_DIRECTORY tests/output)
add_test(NAME TestCBLAS COMMAND cblas_tests)
add_test(NAME TestSqMatrices COMMAND testSQCMatrices tests/input/test.config WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME TestQmle COMMAND LyaPowerEstimate tests/input/test.config WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME NumericalAccuracy COMMAND python tests/compareTestResults.py . WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
