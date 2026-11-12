# Defines the MY_BLAS_INTERFACE target (BLAS + LAPACK + LAPACKE).
# Selected by the USE_OPENBLAS_BREW and USE_MKL_LIB options.
# Included from the top-level CMakeLists.txt; runs in its scope.

include(CheckFunctionExists)
include(CMakePushCheckState)

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
    add_definitions(-DUSE_MKL)
    find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
    message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

    target_compile_options(MY_BLAS_INTERFACE INTERFACE
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(MY_BLAS_INTERFACE INTERFACE
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(MY_BLAS_INTERFACE INTERFACE $<LINK_ONLY:MKL::MKL>)
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(MY_LAPACKE_LIB "")

    # FindLAPACK does not cover the C interface (LAPACKE). Some builds
    # (e.g. Homebrew/conda OpenBLAS) bundle it; Debian/Ubuntu ship it separately.
    cmake_push_check_state(RESET)
        set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
        check_function_exists(LAPACKE_dsyevr LAPACK_HAS_LAPACKE)
    cmake_pop_check_state()

    if (NOT LAPACK_HAS_LAPACKE)
        find_library(LAPACKE_LIBRARY NAMES lapacke REQUIRED)
        find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h
                  PATH_SUFFIXES openblas lapacke)
        message(STATUS "Found LAPACKE: ${LAPACKE_LIBRARY}")
        set(MY_LAPACKE_LIB ${LAPACKE_LIBRARY})
        if (LAPACKE_INCLUDE_DIR)
            target_include_directories(MY_BLAS_INTERFACE INTERFACE ${LAPACKE_INCLUDE_DIR})
        endif()
    endif()

    # lapacke must precede LAPACK on the link line
    target_link_libraries(MY_BLAS_INTERFACE INTERFACE
        ${MY_LAPACKE_LIB} LAPACK::LAPACK BLAS::BLAS)
endif()
