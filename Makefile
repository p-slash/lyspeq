#---------------------------------------
# Set options
#---------------------------------------
# Can interpolate using linear or cubic spline.
# OPT += -DINTERP_1D_TYPE=GSL_LINEAR_INTERPOLATION
OPT += -DINTERP_1D_TYPE=GSL_CUBIC_INTERPOLATION

# OPT += -DINTERP_2D_TYPE=GSL_BILINEAR_INTERPOLATION
OPT += -DINTERP_2D_TYPE=GSL_BICUBIC_INTERPOLATION

# Pick the fit function
# OPT += -DDEBUG_FIT_FUNCTION
OPT += -DPD13_FIT_FUNCTION

# Pick the redshift binning function 
# OPT += -DTOPHAT_Z_BINNING_FN
OPT += -DTRIANGLE_Z_BINNING_FN

# Print out at debugging checkpoints
# OPT += -DDEBUG_ON

#---------------------------------------
# Choose compiler and library options
#---------------------------------------

#SYSTYPE="GNU_XE18MKL"
#SYSTYOE="GNU_ATLAS"
#SYSTYPE="XE18_icpcMKL"
SYSTYPE="MACOSX_clang"
#SYSTYPE="MACOSX_g++"

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
CPPFLAGS= -fopenmp -m64 -I${GSL_DIR}/include -I${MKLROOT}/include
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
# openblas is keg-only, which means it was not symlinked into /usr/local,
# because macOS provides BLAS and LAPACK in the Accelerate framework.
ifeq ($(SYSTYPE),"MACOSX_clang") 
CXX     = clang++ 
CPPFLAGS= -Xpreprocessor -fopenmp -I/usr/local/opt/openblas/include
LDFLAGS = -L/usr/local/opt/openblas/lib
LDLIBS  = -lgsl -lopenblas -lomp
endif

# Does not work!
ifeq ($(SYSTYPE),"MACOSX_g++") 
CXX = g++-9 -fopenmp
INCLS = -I/usr/local/include -I/usr/local/opt/openblas/include
LIBSS = -L/usr/local/lib -L/usr/local/opt/openblas/lib
LINKS = -lgsl -lopenblas -lgomp
endif

# removed flags: -ansi -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wnested-externs -Dinline=
GSLRECFLAGS = -Wall -pedantic -Werror -W \
				-Wshadow -Wpointer-arith \
				-Wcast-qual -Wcast-align -Wwrite-strings \
				-fshort-enums -fno-common \
				-g -O3
				
#---------------------------------------

DEPDIR = dep
DIRS = core io gsltools
VPATH = core:io:gsltools
SRCEXT = cpp

SOURCES = $(shell find $(DIRS) -type f -name '*.$(SRCEXT)')
OBJECTS = $(patsubst %, %.o, $(basename $(SOURCES)))

PROGS = $(shell find . -type f -name '*.$(SRCEXT)')

# -DHAVE_INLINE for inline declarations in GSL for faster performance
CPPFLAGS += -std=gnu++11 -DHAVE_INLINE $(OPT)
CXXFLAGS = $(GSLRECFLAGS)
#$(CXX) $(CPPFLAGS) $(CXXFLAGS)
	
all: LyaPowerEstimate CreateSQLookUpTable cblas_tests

LyaPowerEstimate: LyaPowerEstimate.o $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

CreateSQLookUpTable: CreateSQLookUpTable.o $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

cblas_tests: cblas_tests.o $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(DEPDIR)/%.d: %.cpp 
	@mkdir -p $(DEPDIR)
	$(CXX) -MM -MG $(CPPFLAGS) $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

-include $(DEPDIR)/LyaPowerEstimate.d
-include $(DEPDIR)/CreateSQLookUpTable.d
-include $(DEPDIR)/cblas_tests.d

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
	$(RM) -r $(DEPDIR)
	$(RM) $(addsuffix /*.o, $(DIRS))
	$(RM) LyaPowerEstimate LyaPowerEstimate.o
	$(RM) CreateSQLookUpTable CreateSQLookUpTable.o
	$(RM) test cblas_tests.o


