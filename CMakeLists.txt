cmake_minimum_required(VERSION 3.18)
enable_testing()
project(lyspeq)

option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_MPI "Enable MPI" ON)
option(USE_OPENBLAS_BREW "Use OpenBlas brew" OFF)
option(USE_MKL_LIB "Use MKL library" OFF)
option(DEBUG "Compile with debugging options" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(intel_like_cxx "$<COMPILE_LANG_AND_ID:CXX,Intel,IntelLLVM>")

# message(${CMAKE_SYSTEM})
# if (APPLE)
#     message(aplle)
# endif()

add_compile_options(-std=c++20)
add_library(MY_OMP_MPI INTERFACE)

if (DEBUG)
    add_compile_options(-g)
    add_definitions(-DDEBUG)
else()
    add_compile_options(
        "$<${gcc_like_cxx}:-O3;-march=native>"
        "$<${intel_like_cxx}:-O3;-xHost>"
    )
endif()

find_package(PkgConfig REQUIRED)
if (ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DENABLE_OMP)
    target_link_libraries(MY_OMP_MPI INTERFACE OpenMP::OpenMP_CXX)
endif()

find_package(GSL REQUIRED)
# GSL inline definition
add_definitions(-DHAVE_INLINE)
set_property(TARGET GSL::gsl PROPERTY INTERFACE_LINK_LIBRARIES "")

# =======================
# Create a BLAS interface 
# =======================

add_library(MY_BLAS_INTERFACE INTERFACE)
if (USE_OPENBLAS_BREW)
    # set(ENV{CMAKE_PREFIX_PATH} "/opt/homebrew/opt/openblas")
    # set(BLA_VENDOR OpenBLAS)
    set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/opt/openblas/lib/pkgconfig")
    pkg_check_modules(BLAS REQUIRED openblas)

    target_compile_options(MY_BLAS_INTERFACE INTERFACE ${BLAS_CFLAGS})
    target_link_directories(MY_BLAS_INTERFACE INTERFACE ${BLAS_LIBRARY_DIRS})
    target_link_libraries(MY_BLAS_INTERFACE INTERFACE ${BLAS_LIBRARIES})
elseif (USE_MKL_LIB)
    add_definitions(-DUSE_MKL_CBLAS)
    find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
    message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

    target_compile_options(MY_BLAS_INTERFACE INTERFACE
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(MY_BLAS_INTERFACE INTERFACE
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(MY_BLAS_INTERFACE INTERFACE $<LINK_ONLY:MKL::MKL>)
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(MY_BLAS_INTERFACE INTERFACE BLAS::BLAS)
endif()

# ===
# MPI
# ===
if (ENABLE_MPI)
    add_definitions(-DENABLE_MPI)
    # pkg_search_module(MPI REQUIRED mvapich mpich mpi ompi)
    # set(COMMAND_EXE "mpirun -np 1 ")
    # set(MPI_SKIP_COMPILER_WRAPPER ON)
    find_package(MPI REQUIRED)
    # message(${MPI_VERSION})
    # message("com opt" ${MPI_COMPILE_OPTIONS})
    # message("include dir" ${MPI_INCLUDE_DIRS})
    # message("lib" ${MPI_LIBRARIES})
    # message("flags" ${MPI_CFLAGS})
    # set(MPI_LINKS_MINE mpi)
    target_link_libraries(MY_OMP_MPI INTERFACE MPI::MPI_CXX)
endif()

pkg_check_modules(CFITSIO REQUIRED cfitsio)
add_library(MY_FITSIO_INTERFACE INTERFACE)
target_compile_options(MY_FITSIO_INTERFACE INTERFACE ${CFITSIO_CFLAGS})
target_link_directories(MY_FITSIO_INTERFACE INTERFACE ${CFITSIO_LIBRARY_DIRS})
target_link_libraries(MY_FITSIO_INTERFACE INTERFACE ${CFITSIO_LIBRARIES})

pkg_check_modules(FFTW3 REQUIRED fftw3)
add_library(MY_FFTW3_INTERFACE INTERFACE)
target_compile_options(MY_FFTW3_INTERFACE INTERFACE ${FFTW3_CFLAGS})
target_link_directories(MY_FFTW3_INTERFACE INTERFACE ${FFTW3_LIBRARY_DIRS})
target_link_libraries(MY_FFTW3_INTERFACE INTERFACE ${FFTW3_LIBRARIES})

# =======================
# Compile options for all
# =======================
include_directories(.)

# ================
# Object libraries
# ================

add_library(GlobalObjLibs OBJECT
    core/global_numbers.cpp io/config_file.cpp io/logger.cpp
    io/io_helper_functions.cpp
)

add_library(CoreObjLibs OBJECT
    mathtools/matrix_helper.cpp
    core/one_qso_estimate.cpp
    core/quadratic_estimate.cpp
    io/qso_file.cpp
    mathtools/real_field.cpp
    core/chunk_estimate.cpp
    mathtools/smoother.cpp
    mathtools/bspline.cpp
    io/bootstrap_file.cpp
    mathtools/stats.cpp
)
target_link_libraries(CoreObjLibs PUBLIC
    MY_OMP_MPI GSL::gsl MY_BLAS_INTERFACE MY_FITSIO_INTERFACE
    MY_FFTW3_INTERFACE)

add_library(SqTableObjLibs OBJECT
    core/sq_table.cpp io/sq_lookup_table_file.cpp
    core/fiducial_cosmology.cpp mathtools/fourier_integrator.cpp
    mathtools/discrete_interpolation.cpp mathtools/interpolation.cpp
    mathtools/interpolation_2d.cpp
)
target_link_libraries(SqTableObjLibs PUBLIC MY_OMP_MPI GSL::gsl)

add_library(AllObjLibs INTERFACE)
target_sources(AllObjLibs INTERFACE
    $<TARGET_OBJECTS:GlobalObjLibs> $<TARGET_OBJECTS:CoreObjLibs>
    $<TARGET_OBJECTS:SqTableObjLibs>)
target_link_libraries(AllObjLibs INTERFACE
    MY_OMP_MPI GSL::gsl MY_BLAS_INTERFACE MY_FITSIO_INTERFACE
    MY_FFTW3_INTERFACE)

# custom target to test filtering. Usage: cmake --build . --target print_genex
add_custom_target(print_genex
    COMMAND ${CMAKE_COMMAND} -E echo 'Objects $<FILTER:$<TARGET_OBJECTS:CoreObjLibs>,EXCLUDE,/quadratic_estimate.cpp.o>'
)

# ===========
# Executables
# ===========

add_executable(LyaPowerEstimate LyaPowerEstimate.cpp)
target_link_libraries(LyaPowerEstimate PRIVATE AllObjLibs)

add_executable(LyaPowerxQmlExposure
    LyaPowerxQmlExposure.cpp cross/cross_exposure.cpp cross/one_qso_exposures.cpp)
target_link_libraries(LyaPowerxQmlExposure PRIVATE AllObjLibs)

add_executable(CreateSQLookUpTable CreateSQLookUpTable.cpp)
target_link_libraries(CreateSQLookUpTable PRIVATE GlobalObjLibs SqTableObjLibs)

add_executable(lyspeqBootStats lyspeqBootStats.cpp)
#    $<FILTER:$<TARGET_OBJECTS:CoreObjLibs>,EXCLUDE,/quadratic_estimate.cpp.o>)
target_link_libraries(lyspeqBootStats PRIVATE AllObjLibs)

## ---------
## Test exes
## ---------

add_executable(cblas_tests
    tests/cblas_tests.cpp tests/test_utils.cpp
    mathtools/matrix_helper.cpp mathtools/real_field.cpp
    mathtools/discrete_interpolation.cpp
)
target_compile_definitions(cblas_tests PRIVATE SRCDIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(cblas_tests PUBLIC
    OpenMP::OpenMP_CXX GSL::gsl MY_BLAS_INTERFACE MY_FFTW3_INTERFACE)

add_executable(testSQCMatrices tests/testSQCMatrices.cpp tests/test_utils.cpp)
target_link_libraries(testSQCMatrices PRIVATE AllObjLibs)

# ===========
# Installation
# ===========
install(TARGETS LyaPowerEstimate LyaPowerxQmlExposure CreateSQLookUpTable lyspeqBootStats
        RUNTIME DESTINATION bin)
install(PROGRAMS py/smbivspline.py DESTINATION bin)

# ===========
# Testing
# ===========
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/output)
add_test(NAME TestCBLAS COMMAND cblas_tests)
add_test(NAME TestSqMatrices COMMAND testSQCMatrices tests/input/test.config WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME TestQmle COMMAND LyaPowerEstimate tests/input/test.config WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME NumericalAccuracy COMMAND python tests/compareTestResults.py . WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
