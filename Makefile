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
OPT += -DTOPHAT_Z_BINNING_FN
# OPT += -DTRIANGLE_Z_BINNING_FN

#--------------------------------------- Select target computer

#SYSTYPE="GRACE"
SYSTYPE="LAPTOP"

#--------------------------------------- Adjust settings for target computer

ifeq ($(SYSTYPE),"GRACE")   
GSL_INCL =  -I${GSL_DIR}/include
GSL_LIBS =  -L${GSL_DIR}/lib
endif

ifeq ($(SYSTYPE),"LAPTOP")   
GSL_INCL =  -I/usr/local/include
GSL_LIBS =  -L/usr/local/lib
endif

#---------------------------------------

CXX := g++

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

CPPFLAGS := -std=gnu++11 -Wall -pedantic -Wno-long-long $(GSL_INCL) $(OPT)
LDLIBS := -lfftw3 -lgsl -lgslcblas $(GSL_LIBS)
	
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
