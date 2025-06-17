The sht.c code can be used in examples as follows:

gcc sht_demo.c sht.c -lm -o sht_demo.x
g++ sht_demo.cpp sht.c -lm -o sht_demo.xx

with the usual header file

#include "sht.h"

Options you may want:

-DN_CHECKVAL: adds checking to the spherical harmonic routines

-DTHREAD_SAFE_SHT: avoids static variables (some functions, not likely to be useful here, are not included in this option)
