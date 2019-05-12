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
CXX := g++ -fopenmp -m64
INCLS = -I${GSL_DIR}/include -I${MKLROOT}/include
LIBSS = -L${GSL_DIR}/lib -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed
LINKS = -lgsl -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
endif

# ATLAS cblas
# Has not been tested!
ifeq ($(SYSTYPE),"GNU_ATLAS") 
CXX := g++ -fopenmp 
INCLS = -I${GSL_DIR}/include
LIBSS = -L${GSL_DIR}/lib -L${ATLAS_DIR}/lib
LINKS = -lgsl -lcblas -latlas -lm -ldl
endif

# Parallel Studio XE 2018
# Linux, Intel compiler, Intel(R) 64 arch
# Uses 32-bit integers interface even though 64-bit integers interface works
# OpenMP threading with Intel
# Dynamic linking, no explicit MKL lib linking
# Static linking fails for unknown reasons
ifeq ($(SYSTYPE),"XE18_icpcMKL") 
CXX := icpc -qopenmp -mkl=parallel
INCLS = -I${GSL_DIR}/include
LIBSS = -L${GSL_DIR}/lib
LINKS = -lgsl -liomp5 -lpthread -lm -ldl
endif

# GSL for blas
ifeq ($(SYSTYPE),"MACOSX_clang") 
CXX := clang++ -Xpreprocessor -fopenmp
INCLS = -I/usr/local/include -I/usr/local/opt/libomp/include -I/usr/local/opt/openblas/include
LIBSS = -L/usr/local/lib -L/usr/local/opt/libomp/lib -L/usr/local/opt/openblas/lib
LINKS = -lgsl -lopenblas -lomp
endif

# Does not work!
ifeq ($(SYSTYPE),"MACOSX_g++") 
CXX := g++-9 -fopenmp
INCLS = -I/usr/local/include -I/usr/local/opt/openblas/include
LIBSS = -L/usr/local/lib -L/usr/local/opt/openblas/lib
LINKS = -lgsl -lopenblas -lgomp
endif

# removed flags: -ansi -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wnested-externs -Dinline=
GSLRECFLAGS := -Wall -pedantic -Werror -W \
				-Wshadow -Wpointer-arith \
				-Wcast-qual -Wcast-align -Wwrite-strings \
				-fshort-enums -fno-common \
				-g -O3
				
#---------------------------------------

DEPDIR := dep
COREDIR := core
IODIR := io
GSLTOOLSDIR := gsltools

SRCEXT := cpp

CORESOURCES := $(shell find $(COREDIR) -type f -name '*.$(SRCEXT)')
COREOBJECTS := $(patsubst %, %.o, $(basename $(CORESOURCES)))

GSLTOOLSSOURCES := $(shell find $(GSLTOOLSDIR) -type f -name '*.$(SRCEXT)')
GSLTOOLSOBJECTS := $(patsubst %, %.o, $(basename $(GSLTOOLSSOURCES)))

IOSOURCES := $(shell find $(IODIR) -type f -name '*.$(SRCEXT)')
IOOBJECTS := $(patsubst %, %.o, $(basename $(IOSOURCES)))

# -DHAVE_INLINE for inline declarations in GSL for faster performance
CPPFLAGS := -std=gnu++11 $(GSLRECFLAGS) -DHAVE_INLINE $(OPT) $(INCLS)
LDLIBS := $(LIBSS) $(LINKS)
	
all: LyaPowerEstimate CreateSQLookUpTable
	
LyaPowerEstimate: LyaPowerEstimate.o $(COREOBJECTS) $(GSLTOOLSOBJECTS) $(IOOBJECTS)
	$(CXX) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

CreateSQLookUpTable: CreateSQLookUpTable.o $(COREOBJECTS) $(GSLTOOLSOBJECTS) $(IOOBJECTS)
	$(CXX) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

test: cblas_tests.o $(COREOBJECTS) $(GSLTOOLSOBJECTS) $(IOOBJECTS)
	$(CXX) $(CPPFLAGS) $^ -o $@ $(LDLIBS)

$(DEPDIR)/%.d: %.cpp 
	@mkdir -p $(DEPDIR)
	$(CXX) -MM -MG $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

include $(COREDIR)/core.make
include $(GSLTOOLSDIR)/gsltools.make
include $(IODIR)/io.make

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
	$(RM) $(addsuffix /*.o, $(COREDIR) $(GSLTOOLSDIR) $(IODIR))
	$(RM) LyaPowerEstimate LyaPowerEstimate.o
	$(RM) CreateSQLookUpTable CreateSQLookUpTable.o
	$(RM) test cblas_tests.o


