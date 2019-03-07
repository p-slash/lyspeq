#---------------------------------------
# Set options
#---------------------------------------
# Can interpolate using linear or cubic spline.
OPT += -DINTERP_1D_TYPE=GSL_CUBIC_INTERPOLATION
# OPT += -DINTERP_1D_TYPE=GSL_LINEAR_INTERPOLATION

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
#--------------------------------------- Select target computer

#SYSTYPE="GRACE"
#SYSTYPE="GRACEicc"
SYSTYPE="LAPTOP"

#--------------------------------------- Adjust settings for target computer

ifeq ($(SYSTYPE),"GRACEicc") 
CXX := icpc -DMKL_ILP64 -mkl=parallel 
GSL_INCL = -I${GSL_DIR}/include -I${MKLROOT}/include
GSL_LIBS = -L${GSL_DIR}/lib -L${MKLROOT}/lib/intel64 -lgsl
OMP_FLAG = -openmp
OMP_INCL =
OMP_LIBS = -liomp5 -lpthread -lm -ldl
endif

ifeq ($(SYSTYPE),"GRACE") 
CXX := g++
GSL_INCL = -I${GSL_DIR}/include
GSL_LIBS = -lgsl -lgslcblas -L${GSL_DIR}/lib
OMP_FLAG = -fopenmp
OMP_INCL =
OMP_LIBS =
endif

ifeq ($(SYSTYPE),"LAPTOP") 
CXX := clang++
GSL_INCL = -I/usr/local/opt/gsl/include
GSL_LIBS = -lgsl -lgslcblas -L/usr/local/opt/gsl/lib
OMP_FLAG = -Xpreprocessor -fopenmp
OMP_INCL = -I/usr/local/opt/libomp/include
OMP_LIBS = -lomp -L/usr/local/opt/libomp/lib
endif

# removed flags: -ansi -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wnested-externs -Dinline=
GSLRECFLAGS :=  -Wall -pedantic -Werror -W \
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

CPPFLAGS := $(OMP_FLAG) -std=gnu++11 $(GSLRECFLAGS) $(GSL_INCL) $(OMP_INCL) $(OPT)
LDLIBS := $(GSL_LIBS) $(OMP_LIBS) 
	
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
