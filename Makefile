CXX=g++-9
CPPFLAGS=-fopenmp -I/opt/local/include
LDFLAGS=-L/opt/local/lib -L/usr/local/opt/openblas/lib
LDLIBS=-lgsl -lopenblas -lgomp -lm -ldl
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


