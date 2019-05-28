#---------------------------------------
# Set options
#---------------------------------------
# Can interpolate using linear or cubic spline.
# OPT += -DINTERP_1D_TYPE=GSL_LINEAR_INTERPOLATION
OPT += -DINTERP_1D_TYPE=GSL_CUBIC_INTERPOLATION

# OPT += -DINTERP_2D_TYPE=GSL_BILINEAR_INTERPOLATION
OPT += -DINTERP_2D_TYPE=GSL_BICUBIC_INTERPOLATION

# Pick the fit function
OPT += -DPD13_FIT_FUNCTION
# OPT += -DSOME_OTHER_FIT

# Pick the redshift binning function 
# OPT += -DTOPHAT_Z_BINNING_FN
OPT += -DTRIANGLE_Z_BINNING_FN

# To add a wide high k bin, uncomment and set in km/s
OPT += -DLAST_K_EDGE=10.

#---------------------------------------
# Choose compiler and library options
#---------------------------------------

#SYSTYPE="GNU_XE18MKL"
#SYSTYOE="GNU_ATLAS"
#SYSTYPE="XE18_icpcMKL"
SYSTYPE="clang_openblas"

# List of compiler options
#---------------------------------------

# Parallel Studio XE 2018, MKL for cblas
# Linux, GNU compiler, Intel(R) 64 arch
# 32-bit integers interface, 64-bit interface has runtime errors on Grace
# OpenMP threading with GNU
# Dynamic linking, explicit MKL lib linking
# Compiles with GCC 7.3.0
ifeq ($(SYSTYPE),"GNU_XE18MKL") 
CXX     = g++ 
CPPFLAGS= -fopenmp -m64 -I${GSL_DIR}/include
LDFLAGS = -L${GSL_DIR}/lib -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed
LDLIBS  = -lgsl -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
endif

# ATLAS cblas
# Has not been tested!
ifeq ($(SYSTYPE),"GNU_ATLAS") 
CXX     = g++ 
CPPFLAGS= -fopenmp -I${GSL_DIR}/include 
LDFLAGS = -L${GSL_DIR}/lib -L${ATLAS_DIR}/lib
LDLIBS  = -lgsl -latlas -lm -ldl
endif

# Parallel Studio XE 2018
# Linux, Intel compiler, Intel(R) 64 arch
# Uses 32-bit integers interface even though 64-bit integers interface works
# OpenMP threading with Intel
# Dynamic linking, no explicit MKL lib linking
# Static linking fails for unknown reasons
ifeq ($(SYSTYPE),"XE18_icpcMKL") 
CXX     = icpc 
CPPFLAGS= -qopenmp -mkl=parallel -I${GSL_DIR}/include
LDFLAGS = -L${GSL_DIR}/lib
LDLIBS  = -lgsl -liomp5 -lpthread -lm -ldl
endif

# OpenBLAS 
# To install OpenMP in Mac: brew install libomp
# To install OpenBLAS:      brew install openblas
# openblas is keg-only, which means it was not symlinked into /usr/local,
# because macOS provides BLAS and LAPACK in the Accelerate framework.
ifeq ($(SYSTYPE),"clang_openblas") 
CXX     = clang++ 
CPPFLAGS= -Xpreprocessor -fopenmp
LDFLAGS = -L/usr/local/opt/openblas/lib
LDLIBS  = -lgsl -lopenblas -lomp
endif

# Does not work!
# gcc does not search usual dirs in Mojave
ifeq ($(SYSTYPE),"MACOSX_g++") 
CXX     = g++-9
CPPFLAGS= -fopenmp -I/usr/local/opt/openblas/include
LDFLAGS = -L/usr/local/opt/openblas/lib
LDLIBS  = -lgsl -lopenblas -lgomp
endif
#---------------------------------------

# removed flags: -ansi -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wnested-externs -Dinline=
GSLRECFLAGS = -Wall -pedantic -Werror -W \
				-Wshadow -Wpointer-arith \
				-Wcast-qual -Wcast-align -Wwrite-strings \
				-fshort-enums -fno-common \
				-g -O3
				
#---------------------------------------

DIRS   = core io gsltools
SRCEXT = cpp
vpath  $(DIRS)

SOURCES = $(shell find $(DIRS) -type f -name '*.$(SRCEXT)')
OBJECTS = $(SOURCES:.$(SRCEXT)=.o)
PROGS   = $(basename $(shell find . -type f -name '*.$(SRCEXT)'))

# -DHAVE_INLINE for inline declarations in GSL for faster performance

CPPFLAGS += $(DEPFLAGS) -std=gnu++11 -DHAVE_INLINE $(OPT)
CXXFLAGS = $(GSLRECFLAGS)
	
all: LyaPowerEstimate CreateSQLookUpTable cblas_tests

LyaPowerEstimate: LyaPowerEstimate.o $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

CreateSQLookUpTable: CreateSQLookUpTable.o $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

cblas_tests: cblas_tests.o core/matrix_helper.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

.PHONY: install
install: LyaPowerEstimate CreateSQLookUpTable
	mkdir -p $(HOME)/bin
	cp LyaPowerEstimate CreateSQLookUpTable py/lorentzian_fit.py $(HOME)/bin
	chmod a+x $(HOME)/bin/lorentzian_fit.py

.PHONY: uninstall
uninstall:
	$(RM) $(HOME)/bin/lorentzian_fit.py
	$(RM) $(HOME)/bin/LyaPowerEstimate $(HOME)/bin/CreateSQLookUpTable

.PHONY: clean
clean:
# 	$(RM) -r $(DEPDIR)
	$(RM) $(addsuffix /*.o, $(DIRS))
	$(RM) LyaPowerEstimate LyaPowerEstimate.o
	$(RM) CreateSQLookUpTable CreateSQLookUpTable.o
	$(RM) cblas_tests cblas_tests.o


