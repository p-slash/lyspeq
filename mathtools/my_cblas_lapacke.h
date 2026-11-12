// prevent gsl_cblas.h from being included
#define  __GSL_CBLAS_H__

#if defined(USE_MKL)
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#elif defined(USE_ACCELERATE)
#include "Accelerate/Accelerate.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif
