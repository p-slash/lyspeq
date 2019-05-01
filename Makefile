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
#SYSTYPE="GRACE"
#SYSTYPE="GRACEicc"
#SYSTYPE="GNU_XE18MKL"
#SYSTYPE="XE18_icpcMKL"
SYSTYPE="LAPTOP"

# List of compiler options
#---------------------------------------
# Works with XE 2015
# MKL for cblas
ifeq ($(SYSTYPE),"GRACEicc") 
CXX := icpc -DMKL_ILP64 -mkl=parallel 
GSL_INCL = -I${GSL_DIR}/include
GSL_LIBS = -L${GSL_DIR}/lib -lgsl
MKL_INCL = -I${MKLROOT}/include
MKL_LIBS = -L${MKLROOT}/lib/intel64 
OMP_FLAG = -openmp
OMP_INCL =
OMP_LIBS = -liomp5 -lpthread -lm -ldl
endif

# Parallel Studio XE 2018, MKL for cblas
# Linux, GNU compiler, Intel(R) 64 arch
# OpenMP threading with GNU
# Dynamic linking, explicit MKL lib linking
# Compiles with GCC 7.3.0, run is under progress runtime error
# loading intel does not help
# Using static linking does not help
ifeq ($(SYSTYPE),"GNU_XE18MKL") 
CXX := g++ -fopenmp -DMKL_ILP64 -m64
INCLS = -I${MKLROOT}/include -I${GSL_DIR}/include 
LIBSS = -L${MKLROOT}/lib/intel64 -L${GSL_DIR}/lib -Wl,--no-as-needed
LINKS = -lgsl -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
endif

# Parallel Studio XE 2018
# Linux, Intel compiler, Intel(R) 64 arch
# OpenMP threading with Intel
# Dynamic linking, no explicit MKL lib linking
# Static linking fails for unknown reasons
ifeq ($(SYSTYPE),"XE18_icpcMKL") 
CXX := icpc -qopenmp -DMKL_ILP64 -mkl=parallel
INCLS = -I${GSL_DIR}/include
LIBSS = -L${GSL_DIR}/lib
LINKS = -lgsl -liomp5 -lpthread -lm -ldl
endif

# GSL for blas
ifeq ($(SYSTYPE),"GRACE") 
CXX := g++
GSL_INCL = -I${GSL_DIR}/include
GSL_LIBS = -lgsl -lgslcblas -L${GSL_DIR}/lib
MKL_INCL =
MKL_LIBS =  
OMP_FLAG = -fopenmp
OMP_INCL =
OMP_LIBS =
endif

# GSL for blas
ifeq ($(SYSTYPE),"LAPTOP") 
CXX := clang++ -Xpreprocessor -fopenmp
INCLS = -I/usr/local/include -I/usr/local/opt/libomp/include
LIBSS = -L/usr/local/lib -L/usr/local/opt/libomp/lib
LINKS = -lgsl -lgslcblas -lomp 
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

clean:
	$(RM) -r $(DEPDIR)
	$(RM) $(addsuffix /*.o, $(COREDIR) $(GSLTOOLSDIR) $(IODIR))
	$(RM) LyaPowerEstimate LyaPowerEstimate.o
	$(RM) CreateSQLookUpTable CreateSQLookUpTable.o

.PHONY: clean
