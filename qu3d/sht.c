#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Mathematical constants */
#define PiOverTwo         1.570796326794896619231322
#define Pi                3.1415926535897932384626433832795
#define TwoPi             6.283185307179586476925286766559
#define FourPi            12.5663706143591729538506
#define OneOverSqrtFourPi 0.079577471545947667884441881686257
#define SqrtTwo           1.4142135623730950488016887242097
#define OneOverTwoPi      0.15915494309189533576888376337251
#define Cos72Deg          0.309016994374947424102293
#define Cos36Deg          0.809016994374947424102293
#define TwoCos72Deg       0.618033988749894848204587

/* Parameters to prevent underflow in the recursion of the spherical
 * harmonics:
 */
#define LAMBDA_MIN (1.0e-25)
#define INV_LAMBDA_MIN (1.0e+25)

/* Parameter to prevent division by zero at the poles */
#define MIN_SIN_THETA (1.0e-5)

/* The beta value, i.e. the ratio of the prefactors for SHT
 * over that for FFT.
 */
#define BETA_FFT_SHT 6.0

#define INTERP_XMIN (1.0e-12) /* This is used in interp_coefs_1 */

/* Maximum number of Woodbury transfer matrices */
#define TNUM_MAX 255

/* Smoothing length for icosahedral kernel */
#define ICOSA_DL 6

/*
 * Structure definitions
 */

/* DOUBLE_MAP
 * *** MAP THAT CAN BE INTERPRETED AS A MATRIX OR A VECTOR
 *
 * the matrix has range matrix[xmin..xmax][ymin..ymax]
 * the vector has range vector[nmin..nmax]
 *
 * The vector and matrix indices are related by:
 * vector[nmin+(ymax-ymin+1)*i+j] == matrix[xmin+i][ymin+j]
 *
 * The vector/matrix must be allocated using the routine:
 * allocate_double_map(&A,xmin,xmax,ymin,ymax,nmin);
 *
 * and de-allocated using
 * deallocate_double_map(&A);
 *
 * Note that the range of (x,y) and the starting n must
 * be passed to allocate_double_map, but this is not
 * necessary for deallocate_double_map since these quantities
 * are already in the structure.
 */

typedef struct {
   long xmin;
   long xmax;
   long ymin;
   long ymax;
   long nmin;
   long nmax;
   double **matrix;
   double *vector;
} DOUBLE_MAP;

/* SPHERE_MODES
 * *** A VECTOR OF SPHERICAL HARMONIC COEFFICIENTS ORGANIZED BY L-VALUE ***
 *
 * the coefs have range coefs[L=0..lmax][-L..L]
 * (yes, this is a non-rectangular matrix)
 *
 * the vector has range [0..lmax^2]
 *
 * The vector and coefs indices are related by:
 * coefs[L][M] == vector[L(L+1) + M]
 *
 * The values are to be filled as follows:
 * a_LM == coefs[L][M] + i * coefs[L][-M] (M>0)
 * a_L0 == coefs[L][0]
 * a_L(-M) == (-1)^M ( coefs[L][M] - i * coefs[L][-M] ) (M>0)
 *
 * While the code will not usually crash if you use an alternate convention,
 * this will cause confusion!!!
 */

typedef struct {
   long lmax;
   long Nmode;
   double **coefs;
   double *vector;
} SPHERE_MODES;

/* SPHERE_PIXEL
 * *** A FILE CONTAINING PIXELIZATION OF THE SKY ***
 *
 * area_flag = do we store pixel areas? (1=yes, 0=no)
 * N = number of pixels
 * theta = theta-coordinate (angle from N pole)
 * phi = phi-coordinate (azimuthal angle)
 * psi = orientation (0 = X to East, Y to North; pi/2 = X to North, Y to West)
 * area = pixel area (steradians)
 */

typedef struct {
   unsigned short int area_flag;
   long N;
   double *theta;
   double *phi;
   double *psi;
   double *area;
} SPHERE_PIXEL;

/* WOODBURY_TRANSFER
 * *** CONTAINS INFO FOR WOODBURY PRECONDITIONER ***
 *
 * WhiteNoiseLevel: white noise per mode (l>lsplit0)
 * sqrt_Cl_prior: sqrt(C_l - C_{lsplit0}) for prior
 * TransferMatrix: the transfer matrix itself!
 * lsplit0: cutoff L
 * Nmodes: number of modes
 */

typedef struct {
   double WhiteNoiseLevel;
   double *sqrt_Cl_prior;
   double **TransferMatrix;
   long lsplit0;
   long Nmodes;
   long nsparse;
   double *element;
} WOODBURY_TRANSFER;

/* ICOSAHEDRAL
 * *** A STRUCTURE FOR TEMPERATURE DATA IN ICOSAHEDRAL PIXELIZATION ***
 *
 * Nside: number of pixels on a side in the icosahedron
 * Npix: total number of pixels
 * data: array of pointers to the data.  The structure is as follows --
 *   data[f][x][y] contains the f-th face (1 through 20), and the (x,y)
 *   pixel where 0<=x, 0<=y, x+y<=Nside.  (The face is a triangle!)
 * vector: the data arranged in a vector format, of length Npix.
 */

typedef struct {
   long Nside;
   long Npix;
   double ***data;
   double *vector;
} ICOSAHEDRAL;

/* IFFT_DATA
 * *** A STRUCTURE FOR STORING DATA FOR INVERSE-CONVOLUTION ***
 *
 * area_eff: effective area vector
 */

typedef struct {
   double *area_eff;
   double **kernel_real, **kernel_imag;
} IFFT_DATA;

/*
 * Contains codes for the following memory-related operations:
 *
 * *** NR ROUTINES (public domain) ***
 * nrerror: Handles memory allocation errors gracefully.
 * dmatrix: Allocates double precision matrices.
 * free_dmatrix: De-allocates double precision matrices.
 * imatrix: Allocates integer matrices.
 * free_imatrix: De-allocates integer matrices.
 * dvector: Allocates double precision vectors.
 * free_dvector: De-allocates double precision vectors.
 * lvector: Allocates long integer vectors.
 * free_lvector: De-allocates long integer vectors.
 * ivector: Allocates integer vectors.
 * free_ivector: De-allocates integer vectors.
 *
 * allocate_double_map: Allocates a double precision map.
 * deallocate_double_map: De-allocates a double precision map.
 * allocate_sphere_modes: Allocates a table of spherical harmonic modes.
 * deallocate_sphere_modes: De-allocates a table of spherical harmonic modes.
 * allocate_sphere_pixel: Allocates a spherical pixelization.
 * deallocate_sphere_pixel: De-allocates a spherical pixelization.
 * allocate_icosahedral: Allocates an icosahedral map.
 * deallocate_icosahedral: De-allocates an icosahedral map.
 */

/* nrerror
 * *** NUMERICAL RECIPES ERROR HANDLER ***
 *
 * This code is in the public domain.
 */

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
     printf("Numerical Recipes run-time error...\n");
     printf("%s\n",error_text);
     printf("...now exiting to system...\n");
     exit(1);
}

/* dmatrix
 * *** ALLOCATES DOUBLE PRECISION MATRICES ***
 *
 * the matrix has range m[nrl..nrh][ncl..nch]
 *
 * This code is in the public domain.
 */

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range               */
/* m[nrl..nrh][ncl..nch]                                       */
/* NR_END has been replaced with its value, 1.                 */
{
   long i,j, nrow=nrh-nrl+1,ncol=nch-ncl+1;
   double **m;

   /* allocate pointers to rows */
   m=(double **) malloc((size_t)((nrow+1)*sizeof(double*)));
   if (!m) nrerror("allocation failure 1 in matrix()");
   m += 1;
   m -= nrl;

   /* allocate rows and set pointers to them */
   m[nrl]=(double *)malloc((size_t)((nrow*ncol+1)*sizeof(double)));
   if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
   m[nrl] += 1;
   m[nrl] -= ncl;

   for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

   /* Sets the newly created matrix to zero */
   for(i=nrl;i<=nrh;i++) for(j=ncl;j<=nch;j++) m[i][j] = 0.;

   /* return pointer to array of pointers to rows */
   return m;
}

/* free_dmatrix
 * *** DE-ALLOCATES DOUBLE PRECISION MATRICES ***
 *
 * the matrix has range m[nrl..nrh][ncl..nch]
 *
 * This code is in the public domain.
 */

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free an double matrix allocated by dmatrix() */
/* replaced NR_END => 1, FREE_ARG => (char *)   */
{
   free((char *) (m[nrl]+ncl-1));
   free((char *) (m+nrl-1));
}

/* imatrix
 * *** ALLOCATES INTEGER MATRICES ***
 *
 * the matrix has range m[nrl..nrh][ncl..nch]
 *
 * This code is in the public domain.
 */

int **imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate an integer matrix with subscript range             */
/* m[nrl..nrh][ncl..nch]                                       */
/* NR_END has been replaced with its value, 1.                 */
{
   long i,j, nrow=nrh-nrl+1,ncol=nch-ncl+1;
   int **m;

   /* allocate pointers to rows */
   m=(int **) malloc((size_t)((nrow+1)*sizeof(int*)));
   if (!m) nrerror("allocation failure 1 in matrix()");
   m += 1;
   m -= nrl;

   /* allocate rows and set pointers to them */
   m[nrl]=(int *)malloc((size_t)((nrow*ncol+1)*sizeof(int)));
   if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
   m[nrl] += 1;
   m[nrl] -= ncl;

   for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

   /* Sets the newly created matrix to zero */
   for(i=nrl;i<=nrh;i++) for(j=ncl;j<=nch;j++) m[i][j] = 0;

   /* return pointer to array of pointers to rows */
   return m;
}

/* free_imatrix
 * *** DE-ALLOCATES INTEGER MATRICES ***
 *
 * the matrix has range m[nrl..nrh][ncl..nch]
 *
 * This code is in the public domain.
 */

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
/* free an integer matrix allocated by imatrix() */
/* replaced NR_END => 1, FREE_ARG => (char *)   */
{
   free((char *) (m[nrl]+ncl-1));
   free((char *) (m+nrl-1));
}
/* End free_imatrix */

/* dvector
 * *** ALLOCATES DOUBLE PRECISION VECTORS ***
 *
 * the vector has range m[nl..nh]
 *
 * This code is in the public domain.
 */

double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
/* replaced macros, as with dmatrix etc.                   */
{
   double *v;
   long i;

   v=(double *)malloc((size_t) ((nh-nl+2)*sizeof(double)));
   if (!v) nrerror("allocation failure in dvector()");

   /* Sets the newly created vector to zero */
   for(i=0;i<nh-nl+2;i++) v[i] = 0.;

   return(v-nl+1);
}
/* End dvector */

/* free_dvector
 * *** DE-ALLOCATES DOUBLE PRECISION VECTORS ***
 *
 * the vector has range m[nl..nh]
 *
 * This code is in the public domain.
 */

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
   free((char*) (v+nl-1));
}
/* End free_dvector */

/* lvector
 * *** ALLOCATES LONG INTEGER VECTORS ***
 *
 * the vector has range m[nl..nh]
 *
 * This code is in the public domain.
 */

long *lvector(long nl, long nh)
/* allocate a long vector with subscript range v[nl..nh] */
{
   long *v;
   long i;

   v=(long *)malloc((size_t) ((nh-nl+2)*sizeof(long)));
   if (!v) nrerror("allocation failure in lvector()");

   /* Sets the newly created vector to zero */
   for(i=0;i<=nh-nl;i++) v[i] = 0;

   return(v-nl+1);
}
/* End lvector */

/* free_lvector
 * *** DE-ALLOCATES LONG INTEGER VECTORS ***
 *
 * the vector has range m[nl..nh]
 *
 * This code is in the public domain.
 */

void free_lvector(long *v, long nl, long nh)
/* free a long vector allocated with lvector() */
{
   free((char*) (v+nl-1));
}
/* End free_lvector */

/* ivector
 * *** ALLOCATES LONG INTEGER VECTORS ***
 *
 * the vector has range m[nl..nh]
 *
 * This code is in the public domain.
 */

int *ivector(long nl, long nh)
/* allocate an integer vector with subscript range v[nl..nh] */
{
   int *v;

   v=(int *)malloc((size_t) ((nh-nl+2)*sizeof(int)));
   if (!v) nrerror("allocation failure in ivector()");
   return(v-nl+1);
}
/* End ivector */

/* free_ivector
 * *** DE-ALLOCATES INTEGER VECTORS ***
 *
 * the vector has range m[nl..nh]
 *
 * This code is in the public domain.
 */

void free_ivector(int *v, long nl, long nh)
/* free an integer vector allocated with ivector() */
{
   free((char*) (v+nl-1));
}
/* End free_ivector */

/* NUMERICAL RECIPES NON-PUBLIC DOMAIN FUNCTION */

/* gaussjinv
 * *** INVERTS A MATRIX ***
 *
 * Takes matrix A[0..n-1][0..n-1] and inverts it.  The original matrix A is destroyed.
 *
 * Arguments:
 * > A: matrix to be inverted
 *   n: dimension of matrix
 */

void gaussjinv(double **A, int n) {
   /* This routine inverts a matrix A and stores the result back to
    * A.  The matrices are taken to be n by n.  This routine is a
    * slight modification to Numerical Recipes gaussj, see Section
    * 2.1 p.39 in NR C 2ed.  Note we've replaced float with double.
    */

   int *indxc, *indxr, *ipiv;
   int i,icol,irow,j,k,l,ll;
   double big,dum,pivinv,temp;

   icol = irow = 0; /* These variables have values assigned before they */
                    /* are used, but this avoids the compiler warning.  */

   /* The integer arrays ipiv, indxr, and indxc are used for    */
   /* bookkeeping on the pivoting.                              */
   indxc = ivector(0,n-1);
   indxr = ivector(0,n-1);
   ipiv = ivector(0,n-1);

   for(j=0;j<n;j++) ipiv[j]=0;
   for(i=0;i<n;i++) { /* This is the main loop over the columns */
      big=0.0;                              /* to be reduced. */
      for(j=0;j<n;j++)       /* This is the outer loop of the */
         if (ipiv[j] != 1)   /* search for a pivot element. */
            for (k=0;k<n;k++) {
               if (ipiv[k] == 0) {
                  if (fabs(A[j][k]) >= big) {
                     big=fabs(A[j][k]);
                     irow=j;
                     icol=k;
                  }
               } else if (ipiv[k] > 1)
                  nrerror("gaussj: Singular Matrix-1");
               } /* end for(k) loop */
      /* also end for(j) loop */
      ++(ipiv[icol]);

      /* We now have the pivot element, so we interchange
       * rows, if needed, to put the pivot element on the
       * diagonal.  The columns are not physically
       * interchanged, only relabeled: indxc[i], the column
       * of the ith pivot element, is the ith column that is
       * reduced, while indxr[i] is the row in which that
       * pivot element was originally located.  If indxr[i]
       * != indxc[i] there is an implied column interchange.
       * With this form of bookkeeping, the solution Bs (of
       * which we don't have any!) will end up in the correct
       * order, and the inverse matrix (i.e. that remains in
       * A) will be scrambled by columns.
       */

      if (irow != icol) {
         for (l=0;l<n;l++) {
            /* Swap A[irow][l] and A[icol][l] */
            temp = A[irow][l];
            A[irow][l] = A[icol][l];
            A[icol][l] = temp;
         }
      } /* end if */
      indxr[i]=irow; /* We are now ready to divide the pivot  */
      indxc[i]=icol; /* row by the pivot element, located at  */
                                            /* irow and icol. */
      if (A[icol][icol] == 0.0)
         nrerror("gaussj: Singular Matrix-2");
      pivinv=1.0/A[icol][icol];
      A[icol][icol]=1.0;
      for (l=0;l<n;l++) A[icol][l] *= pivinv;
      for (ll=0;ll<n;ll++) /* Now we reduce the rows, except  */
         if (ll != icol) { /* for the pivot one, of course. */
            dum=A[ll][icol];
            A[ll][icol]=0.0;
            for(l=0;l<n;l++) A[ll][l] -= A[icol][l]*dum;
         }
   }

   /* This is the end of the main loop over columns of the
    * reduction. It only remains to unscramble the solution in
    * view of the column interchanges.  We do this by
    * interchanging pairs of columns in the reverse order that
    * the permutation was built up.
    */
   for (l=n-1;l>=0;l--) {
      if (indxr[l] != indxc[l])
         for (k=0;k<n;k++) {
            temp = A[k][indxr[l]];
            A[k][indxr[l]] = A[k][indxc[l]];
            A[k][indxc[l]] = temp;
         }
   } /* And we are done. */

   /* Clean up memory */
   free_ivector(ipiv,0,n-1);
   free_ivector(indxr,0,n-1);
   free_ivector(indxc,0,n-1);
}

/* OUR FUNCTIONS */

/* allocate_double_map
 * *** ALLOCATES A DOUBLE PRECISION MAP ***
 *
 * the matrix has range matrix[xmin..xmax][ymin..ymax]
 * the vector has range vector[nmin..nmax]
 *
 * The vector and matrix indices are related by:
 * vector[nmin+(ymax-ymin+1)*i+j] == matrix[xmin+i][ymin+j]
 */

void allocate_double_map(DOUBLE_MAP *A, long xmin, long xmax, long ymin,
   long ymax, long nmin) {

   long i;
   long nx, ny;

   /* compute number of rows and columns */
   nx = xmax-xmin+1;
   ny = ymax-ymin+1;

   /* set the index limits */
   A->xmin = xmin;
   A->xmax = xmax;
   A->ymin = ymin;
   A->ymax = ymax;
   A->nmin = nmin;
   A->nmax = nmin + nx*ny - 1;

   /* allocate the memory for the vector */
   A->vector = dvector(nmin,A->nmax);

   /* now allocate the matrix */
   A->matrix = (double**)malloc((size_t)((nx+1)*sizeof(double*)));
   if (!(A->matrix)) {
      printf("Error: Map allocation failure.\n");
      exit(1);
   }
   A->matrix ++;
   A->matrix -= xmin;

   /* matrix allocated, now make it point to the vector */
   for(i=xmin;i<=xmax;i++)
      A->matrix[i] = A->vector + ny*(i-xmin) - ymin;

   /* make sure the vector is initialized to zero */
   for(i=A->nmin;i<=A->nmax;i++) A->vector[i] = 0.;
}

/* deallocate_double_map
 * *** DE-ALLOCATES A DOUBLE PRECISION MAP ***
 *
 * The map should have been allocated by allocate_double_map.
 *
 */

void deallocate_double_map(DOUBLE_MAP *A) {

   /* De-allocate the matrix */
   free((char*)(A->matrix + A->xmin - 1));

   /* De-allocate the vector */
   free_dvector(A->vector,A->nmin,A->nmax);

}

/* allocate_sphere_modes
 * *** ALLOCATES A SPHERICAL MAP ***
 *
 * the coefs have range coefs[L=0..lmax][-L..L]
 * (yes, this is a non-rectangular matrix)
 *
 * the vector has range [0..lmax^2]
 *
 * The vector and coefs indices are related by:
 * coefs[L][M] == vector[L(L+1) + M]
 */

void allocate_sphere_modes(SPHERE_MODES *A, long lmax) {

   long L;

   /* Set the dimension of the system */
   A->lmax = lmax;
   A->Nmode = (lmax+1)*(lmax+1);

   /* Allocate the memory for the data */
   A->vector = dvector(0,A->Nmode-1);

   /* Allocate and set the pointers */
   A->coefs = (double**)malloc((size_t)((lmax+1)*sizeof(double*)));
   for(L=0;L<=lmax;L++)
      A->coefs[L] = A->vector + L*(L+1);

}

/* deallocate_sphere_modes
 * *** DE-ALLOCATES A SPHERICAL MAP ***
 *
 * De-allocates a spherical harmonic map generated by
 * allocate_sphere_modes
 */

void deallocate_sphere_modes(SPHERE_MODES *A) {

   free_dvector(A->vector,0,A->Nmode-1);
   free((char*)(A->coefs));

}

/* allocate_sphere_pixel
 * *** ALLOCATES A SPHERICAL PIXELIZATION ***
 *
 * Arguments:
 * > A: spherical pixelization to be allocated.
 *   N: number of pixels
 *   area_flag: will we want to store pixel area values? (1=yes; 0=no)
 */

void allocate_sphere_pixel(SPHERE_PIXEL *A, long N, unsigned short int area_flag) {

   A->area_flag = area_flag;
   A->N = N;

   /* Allocate memory here */
   A->theta = dvector(0,N-1);
   A->phi = dvector(0,N-1);
   A->psi = dvector(0,N-1);
   if (area_flag) A->area = dvector(0,N-1);
}

/* deallocate_sphere_pixel
 * *** DE-ALLOCATES A SPHERICAL PIXELIZATION ***
 *
 * Arguments:
 * > A: spherical pixelization to be de-allocated.
 */

void deallocate_sphere_pixel(SPHERE_PIXEL *A) {

   long N = A->N;

   free_dvector(A->theta,0,N-1);
   free_dvector(A->phi,0,N-1);
   free_dvector(A->psi,0,N-1);
   if (A->area_flag) free_dvector(A->area,0,N-1);
}

/* allocate_icosahedral
 * *** ALLOCATES AN ICOSAHEDRAL STRUCTURE ***
 *
 * Arguments:
 * > A: icosahedral map to be allocated
 *   Nside: value of Nside to use (approx. Nside^2/2 pixels per face)
 */

void allocate_icosahedral(ICOSAHEDRAL *A, long Nside) {

   long f, x;
   long Npix_per_face = ( (Nside+1)*(Nside+2) ) >>1;
   double *aptr;

#ifdef N_CHECKVAL
   /* Complain if Nside is large enough to overflow the long integer
    * when we count pixels
    */
   if (Nside > 14000) {
      fprintf(stderr, "Error: Nside=%ld too large, this will result in memory error.\n", Nside);
      exit(1);
   }
#endif

   /* Set Nside, build array of double**'s for the faces */
   A->Nside = Nside;
   A->Npix = 20 * Npix_per_face;
   A->vector = aptr = dvector(0, A->Npix-1);
   A->data = (double ***) malloc((size_t)(21 * sizeof(double **)));

   /* Build each face */
   for(f=1;f<=20;f++) {
      A->data[f] = (double **) malloc((size_t)((Nside+1) * sizeof(double *)));
      for(x=0;x<=Nside;x++) {
         A->data[f][x] = aptr;
         aptr += Nside+1-x;
      }
   }
}

/* deallocate_icosahedral
 * *** DE-ALLOCATES AN ICOSAHEDRAL STRUCTURE ***
 *
 * Arguments:
 * > A: icosahedral map to be deallocated
 */

void deallocate_icosahedral(ICOSAHEDRAL *A) {

   long Nside;
   int f;

   Nside = A->Nside;

   /* Disassemble each face */
   for(f=1;f<=20;f++) {
      free((char *)(A->data[f]));
   }
   free((char *)(A->data));

   /* Disassemble data vector */
   free_dvector(A->vector, 0, 10*Nside*(Nside+1)-1);
}

/* fourier_trans_1
 * *** FOURIER TRANSFORMS A DATA SET WITH LENGTH A POWER OF 2 ***
 *
 * This is a Fourier transform routine.  It has the same calling
 * interface as Numerical Recipes four1 and uses the same algorithm
 * as that routine.  This function is slightly faster than NR four1
 * because we have minimized the use of expensive array look-ups.
 *
 * Replaces data[1..2*nn] by its discrete Fourier transform, if
 * isign is input as 1; or replaces data[1..2*nn] by nn times its
 * inverse discrete Fourier transform, if isign is input as -1.
 * data is a complex array of length nn.
 *
 */

void fourier_trans_1(double *data, long nn, int isign) {

   double *data_null;
   double *data_i, *data_i1;
   double *data_j, *data_j1;
   double temp, theta, sintheta, oneminuscostheta;
   double wr, wi, wtemp;
   double tempr1, tempr2, tempr, tempi;

   unsigned long ndata; /* = 2*nn */
   unsigned long lcurrent; /* length of current FFT; will range from 2..n */
   unsigned long i,j,k,m,istep;

   /* Convert to null-offset vector, find number of elements */
   data_null = data + 1;
   ndata = (unsigned long)nn << 1;

   /* Bit reversal */
   data_i = data_null;
   for(i=0;i<nn;i++) {

      /* Here we set data_j equal to data_null plus twice the bit-reverse of i */
      j=0;
      k=i;
      for(m=ndata>>2;m>=1;m>>=1) {
         if (k & 1) j+=m;
         k >>= 1;
      }

      /* If i<j, swap the i and j complex elements of data_null
       * Notice that these are the (2i,2i+1) and (2j,2j+1)
       * real elements.
       */
      if (i<j) {
         data_j = data_null + (j<<1);
         temp = *data_i; *data_i = *data_j; *data_j = temp;
         data_i++; data_j++;
         temp = *data_i; *data_i = *data_j; *data_j = temp;
      } else {
         data_i++;
      }

      /* Now increment data_i so it points to data_null[2i+2]; this is
       * important when we start the next iteration.
       */
      data_i++;
   }

   /* Now do successive FFTs */
   for(lcurrent=2;lcurrent<ndata;lcurrent<<=1) {

      /* Find the angle between successive points and its trig
       * functions, the sine and 1-cos. (Use 1-cos for stability.)
       */
      theta = TwoPi/lcurrent * isign;
      sintheta = sin(theta);
      oneminuscostheta = sin(0.5*theta);
      oneminuscostheta = 2.0*oneminuscostheta*oneminuscostheta;

      /* FFT the individual length-lcurrent segments */
      wr = 1.0;
      wi = 0.0;
      istep = lcurrent<<1;
      for(m=0;m<lcurrent;m+=2) {
         for(i=m;i<ndata;i+=istep) {
            /* Set the data pointers so we don't need to do
             * time-consuming array lookups.
             */
            data_j1=data_j = (data_i1=data_i = data_null + i) + lcurrent;
            data_i1++;
            data_j1++;

            /* Now apply Danielson-Lanczos formula */
            tempr = (tempr1=wr*(*data_j)) - (tempr2=wi*(*data_j1));
            tempi = (wr+wi)*((*data_j)+(*data_j1)) - tempr1 - tempr2;
            /*
             * at this point, tempr + i*tempi is equal to the product of
             * the jth complex array element and w.
             */
            *data_j = (*data_i) - tempr;
            *data_j1 = (*data_i1) - tempi;
            *data_i += tempr;
            *data_i1 += tempi;

         }

         /* Now increment trig recurrence */
         wr -= (wtemp=wr)*oneminuscostheta + wi*sintheta;
         wi += wtemp*sintheta - wi*oneminuscostheta;
      }
   }
}

/* fourier_trans_2
 * *** FOURIER TRANSFORMS A DATA SET OF ARBITRARY LENGTH ***
 *
 * This is a Fourier transform routine.  It has the same calling
 * interface as NR four1 and our fourier_trans_1, but it works for
 * non-power-of-two lengths.
 *
 * This code works by finding the largest factor of 2 dividing nn
 * and doing this portion via FFT.  The odd factors of nn are computed
 * via brute force.  Consequently this routine will run much faster
 * if the length of your data set is a power of 2 times a small odd
 * number.  If your data set has length exactly a power of 2, it is
 * recommended that you use fourier_trans_1 to reduce RAM usage.
 */

void fourier_trans_2(double *data, long nn, int isign) {

   double *data_ptr;
   double *data2;
   long i,j,p,b,nu;
   double oneminuscosphi, sinphi, temp;
   double cexpr, cexpi;
   double *costable, *sintable;
   unsigned long nn2, unsigned_comp_nn, mantissa, two_nn2;
   unsigned long i_in, i_out;
   long phase;

   data2 = dvector(1,2*nn);

   /* Separate nn into mantissa*nn2, where nn2 is a power of 2.
    * We want nn2 to be the largest power of 2 for which the
    * mantissa is an integer.
    */
   unsigned_comp_nn = ~ (unsigned long)nn;
   nn2=1;
   while (nn2 & unsigned_comp_nn) nn2 <<= 1;
   two_nn2 = nn2 << 1;
   mantissa = nn/nn2;

   /* Now re-arrange the data as follows: element mantissa*p+b of data
    * is moved to element nn2*b+p of data2.  Note that we're moving
    * complex numbers and the array indices are doubled accordingly.
    */
   i_in = 0;
   for(p=0;p<nn2;p++) {
      for(b=0;b<mantissa;b++) {
         /* We already have i_in == (mantissa*p+b) << 1 */
         i_out = (nn2*b+p) << 1;
         data2[++i_out] = data[++i_in];
         data2[++i_out] = data[++i_in];
      } /* end b loop */
   } /* end p loop */

   /* Do the FFT of each component of data2 */
   for(b=0;b<mantissa;b++) fourier_trans_1(data2+b*two_nn2,nn2,isign);

   /* Build the trig function tables for the order-mantissa Fourier
    * transform.  costable[i] = cos ( TwoPi/nn * i ).  Similar for
    * sintable.
    */
   oneminuscosphi = sin(Pi/nn * isign);
   oneminuscosphi *= 2*oneminuscosphi;
   sinphi = sin(TwoPi/nn * isign);
   costable = dvector(0,nn-1);
   sintable = dvector(0,nn-1);
   cexpr = 1.;
   cexpi = 0.;
   for(i=0;i<nn;i++) {
      costable[i] = cexpr;
      sintable[i] = cexpi;
      cexpr -= oneminuscosphi * (temp=cexpr) + sinphi * cexpi;
      cexpi += sinphi * temp - oneminuscosphi * cexpi;
   }

   /* It remains to do the order-mantissa Fourier transform to
    * complete the operation.  We do this by clearing the input
    * array and doing the order-mantissa Fourier transform from
    * data2 to data.
    */
   data_ptr = data;
   for(i=0;i<nn;i++) {
      data_ptr[1] = data_ptr[2] = 0.;
      nu = 2*(i % nn2);
      phase = 0; /* phase will be (i*j)%nn */
      for(j=0;j<mantissa;j++) {
         data_ptr[1] += costable[phase] * data2[nu+1] - sintable[phase] * data2[nu+2];
         data_ptr[2] += sintable[phase] * data2[nu+1] + costable[phase] * data2[nu+2];
         if ((phase += i) >= nn) phase -= nn;
         nu += two_nn2;
      } /* end j loop */

      data_ptr += 2;
   } /* end i loop */

   /* Clean up memory */
   free_dvector(data2,1,2*nn);
   free_dvector(costable,0,nn-1);
   free_dvector(sintable,0,nn-1);
}

/* interp_coefs_1
 * *** FINDS THE COEFFICIENTS FOR A POLYNOMIAL INTERPOLATION ***
 *
 * This routine computes the coefficients c_i for determining
 * y(x) given y evaluated at -M, -M+1, ... N
 * using an interpolating polynomial of order M+N.  We require x to
 * be in the range [0,1].  The formula for y(x) is:
 *
 * y(x) = sum_i c[i] y(i), i=-M .. N.
 */

void interp_coefs_1(double x, int M, int N, double *c) {

   int i,j;
   double product;

#define MAX_FACTORIAL 21
#ifdef THREAD_SAFE_SHT
   double factorial_products[MAX_FACTORIAL][MAX_FACTORIAL] = {
     {   1.000000000000000E+00,   1.000000000000000E+00,   2.000000000000000E+00,   6.000000000000000E+00,   2.400000000000000E+01,   1.200000000000000E+02, 
         7.200000000000000E+02,   5.040000000000000E+03,   4.032000000000000E+04,   3.628800000000000E+05,   3.628800000000000E+06,   3.991680000000000E+07, 
         4.790016000000000E+08,   6.227020800000000E+09,   8.717829120000000E+10,   1.307674368000000E+12,   2.092278988800000E+13,   3.556874280960000E+14, 
         6.402373705728000E+15,   1.216451004088320E+17,   2.432902008176640E+18},
     {   1.000000000000000E+00,   1.000000000000000E+00,   2.000000000000000E+00,   6.000000000000000E+00,   2.400000000000000E+01,   1.200000000000000E+02, 
         7.200000000000000E+02,   5.040000000000000E+03,   4.032000000000000E+04,   3.628800000000000E+05,   3.628800000000000E+06,   3.991680000000000E+07, 
         4.790016000000000E+08,   6.227020800000000E+09,   8.717829120000000E+10,   1.307674368000000E+12,   2.092278988800000E+13,   3.556874280960000E+14, 
         6.402373705728000E+15,   1.216451004088320E+17,   2.432902008176640E+18},
     {   2.000000000000000E+00,   2.000000000000000E+00,   4.000000000000000E+00,   1.200000000000000E+01,   4.800000000000000E+01,   2.400000000000000E+02, 
         1.440000000000000E+03,   1.008000000000000E+04,   8.064000000000000E+04,   7.257600000000000E+05,   7.257600000000000E+06,   7.983360000000000E+07, 
         9.580032000000000E+08,   1.245404160000000E+10,   1.743565824000000E+11,   2.615348736000000E+12,   4.184557977600000E+13,   7.113748561920000E+14, 
         1.280474741145600E+16,   2.432902008176640E+17,   4.865804016353280E+18},
     {   6.000000000000000E+00,   6.000000000000000E+00,   1.200000000000000E+01,   3.600000000000000E+01,   1.440000000000000E+02,   7.200000000000000E+02, 
         4.320000000000000E+03,   3.024000000000000E+04,   2.419200000000000E+05,   2.177280000000000E+06,   2.177280000000000E+07,   2.395008000000000E+08, 
         2.874009600000000E+09,   3.736212480000000E+10,   5.230697472000000E+11,   7.846046208000000E+12,   1.255367393280000E+14,   2.134124568576000E+15, 
         3.841424223436800E+16,   7.298706024529920E+17,   1.459741204905984E+19},
     {   2.400000000000000E+01,   2.400000000000000E+01,   4.800000000000000E+01,   1.440000000000000E+02,   5.760000000000000E+02,   2.880000000000000E+03, 
         1.728000000000000E+04,   1.209600000000000E+05,   9.676800000000000E+05,   8.709120000000000E+06,   8.709120000000000E+07,   9.580032000000000E+08, 
         1.149603840000000E+10,   1.494484992000000E+11,   2.092278988800000E+12,   3.138418483200000E+13,   5.021469573120000E+14,   8.536498274304000E+15, 
         1.536569689374720E+17,   2.919482409811968E+18,   5.838964819623936E+19},
     {   1.200000000000000E+02,   1.200000000000000E+02,   2.400000000000000E+02,   7.200000000000000E+02,   2.880000000000000E+03,   1.440000000000000E+04, 
         8.640000000000000E+04,   6.048000000000000E+05,   4.838400000000000E+06,   4.354560000000000E+07,   4.354560000000000E+08,   4.790016000000000E+09, 
         5.748019200000000E+10,   7.472424960000000E+11,   1.046139494400000E+13,   1.569209241600000E+14,   2.510734786560000E+15,   4.268249137152000E+16, 
         7.682848446873600E+17,   1.459741204905984E+19,   2.919482409811968E+20},
     {   7.200000000000000E+02,   7.200000000000000E+02,   1.440000000000000E+03,   4.320000000000000E+03,   1.728000000000000E+04,   8.640000000000000E+04, 
         5.184000000000000E+05,   3.628800000000000E+06,   2.903040000000000E+07,   2.612736000000000E+08,   2.612736000000000E+09,   2.874009600000000E+10, 
         3.448811520000000E+11,   4.483454976000000E+12,   6.276836966400000E+13,   9.415255449600000E+14,   1.506440871936000E+16,   2.560949482291200E+17, 
         4.609709068124160E+18,   8.758447229435904E+19,   1.751689445887181E+21},
     {   5.040000000000000E+03,   5.040000000000000E+03,   1.008000000000000E+04,   3.024000000000000E+04,   1.209600000000000E+05,   6.048000000000000E+05, 
         3.628800000000000E+06,   2.540160000000000E+07,   2.032128000000000E+08,   1.828915200000000E+09,   1.828915200000000E+10,   2.011806720000000E+11, 
         2.414168064000000E+12,   3.138418483200000E+13,   4.393785876480000E+14,   6.590678814720000E+15,   1.054508610355200E+17,   1.792664637603840E+18, 
         3.226796347686912E+19,   6.130913060605133E+20,   1.226182612121027E+22},
     {   4.032000000000000E+04,   4.032000000000000E+04,   8.064000000000000E+04,   2.419200000000000E+05,   9.676800000000000E+05,   4.838400000000000E+06, 
         2.903040000000000E+07,   2.032128000000000E+08,   1.625702400000000E+09,   1.463132160000000E+10,   1.463132160000000E+11,   1.609445376000000E+12, 
         1.931334451200000E+13,   2.510734786560000E+14,   3.515028701184000E+15,   5.272543051776000E+16,   8.436068882841600E+17,   1.434131710083072E+19, 
         2.581437078149530E+20,   4.904730448484106E+21,   9.809460896968212E+22},
     {   3.628800000000000E+05,   3.628800000000000E+05,   7.257600000000000E+05,   2.177280000000000E+06,   8.709120000000000E+06,   4.354560000000000E+07, 
         2.612736000000000E+08,   1.828915200000000E+09,   1.463132160000000E+10,   1.316818944000000E+11,   1.316818944000000E+12,   1.448500838400000E+13, 
         1.738201006080000E+14,   2.259661307904000E+15,   3.163525831065600E+16,   4.745288746598400E+17,   7.592461994557440E+18,   1.290718539074765E+20, 
         2.323293370334577E+21,   4.414257403635696E+22,   8.828514807271392E+23},
     {   3.628800000000000E+06,   3.628800000000000E+06,   7.257600000000000E+06,   2.177280000000000E+07,   8.709120000000000E+07,   4.354560000000000E+08, 
         2.612736000000000E+09,   1.828915200000000E+10,   1.463132160000000E+11,   1.316818944000000E+12,   1.316818944000000E+13,   1.448500838400000E+14, 
         1.738201006080000E+15,   2.259661307904000E+16,   3.163525831065600E+17,   4.745288746598400E+18,   7.592461994557440E+19,   1.290718539074765E+21, 
         2.323293370334577E+22,   4.414257403635696E+23,   8.828514807271392E+24},
     {   3.991680000000000E+07,   3.991680000000000E+07,   7.983360000000000E+07,   2.395008000000000E+08,   9.580032000000000E+08,   4.790016000000000E+09, 
         2.874009600000000E+10,   2.011806720000000E+11,   1.609445376000000E+12,   1.448500838400000E+13,   1.448500838400000E+14,   1.593350922240000E+15, 
         1.912021106688000E+16,   2.485627438694400E+17,   3.479878414172160E+18,   5.219817621258240E+19,   8.351708194013184E+20,   1.419790392982241E+22, 
         2.555622707368034E+23,   4.855683143999265E+24,   9.711366287998530E+25},
     {   4.790016000000000E+08,   4.790016000000000E+08,   9.580032000000000E+08,   2.874009600000000E+09,   1.149603840000000E+10,   5.748019200000000E+10, 
         3.448811520000000E+11,   2.414168064000000E+12,   1.931334451200000E+13,   1.738201006080000E+14,   1.738201006080000E+15,   1.912021106688000E+16, 
         2.294425328025600E+17,   2.982752926433280E+18,   4.175854097006592E+19,   6.263781145509888E+20,   1.002204983281582E+22,   1.703748471578690E+23, 
         3.066747248841641E+24,   5.826819772799119E+25,   1.165363954559824E+27},
     {   6.227020800000000E+09,   6.227020800000000E+09,   1.245404160000000E+10,   3.736212480000000E+10,   1.494484992000000E+11,   7.472424960000000E+11, 
         4.483454976000000E+12,   3.138418483200000E+13,   2.510734786560000E+14,   2.259661307904000E+15,   2.259661307904000E+16,   2.485627438694400E+17, 
         2.982752926433280E+18,   3.877578804363264E+19,   5.428610326108570E+20,   8.142915489162854E+21,   1.302866478266057E+23,   2.214873013052296E+24, 
         3.986771423494134E+25,   7.574865704638854E+26,   1.514973140927771E+28},
     {   8.717829120000000E+10,   8.717829120000000E+10,   1.743565824000000E+11,   5.230697472000000E+11,   2.092278988800000E+12,   1.046139494400000E+13, 
         6.276836966400000E+13,   4.393785876480000E+14,   3.515028701184000E+15,   3.163525831065600E+16,   3.163525831065600E+17,   3.479878414172160E+18, 
         4.175854097006592E+19,   5.428610326108570E+20,   7.600054456551997E+21,   1.140008168482800E+23,   1.824013069572479E+24,   3.100822218273215E+25, 
         5.581479992891787E+26,   1.060481198649440E+28,   2.120962397298879E+29},
     {   1.307674368000000E+12,   1.307674368000000E+12,   2.615348736000000E+12,   7.846046208000000E+12,   3.138418483200000E+13,   1.569209241600000E+14, 
         9.415255449600000E+14,   6.590678814720000E+15,   5.272543051776000E+16,   4.745288746598400E+17,   4.745288746598400E+18,   5.219817621258240E+19, 
         6.263781145509888E+20,   8.142915489162854E+21,   1.140008168482800E+23,   1.710012252724199E+24,   2.736019604358719E+25,   4.651233327409823E+26, 
         8.372219989337680E+27,   1.590721797974159E+29,   3.181443595948318E+30},
     {   2.092278988800000E+13,   2.092278988800000E+13,   4.184557977600000E+13,   1.255367393280000E+14,   5.021469573120000E+14,   2.510734786560000E+15, 
         1.506440871936000E+16,   1.054508610355200E+17,   8.436068882841600E+17,   7.592461994557440E+18,   7.592461994557440E+19,   8.351708194013184E+20, 
         1.002204983281582E+22,   1.302866478266057E+23,   1.824013069572479E+24,   2.736019604358719E+25,   4.377631366973951E+26,   7.441973323855716E+27, 
         1.339555198294029E+29,   2.545154876758655E+30,   5.090309753517309E+31},
     {   3.556874280960000E+14,   3.556874280960000E+14,   7.113748561920000E+14,   2.134124568576000E+15,   8.536498274304000E+15,   4.268249137152000E+16, 
         2.560949482291200E+17,   1.792664637603840E+18,   1.434131710083072E+19,   1.290718539074765E+20,   1.290718539074765E+21,   1.419790392982241E+22, 
         1.703748471578690E+23,   2.214873013052296E+24,   3.100822218273215E+25,   4.651233327409822E+26,   7.441973323855715E+27,   1.265135465055472E+29, 
         2.277243837099849E+30,   4.326763290489712E+31,   8.653526580979425E+32},
     {   6.402373705728000E+15,   6.402373705728000E+15,   1.280474741145600E+16,   3.841424223436800E+16,   1.536569689374720E+17,   7.682848446873600E+17, 
         4.609709068124160E+18,   3.226796347686912E+19,   2.581437078149530E+20,   2.323293370334577E+21,   2.323293370334577E+22,   2.555622707368034E+23, 
         3.066747248841641E+24,   3.986771423494134E+25,   5.581479992891787E+26,   8.372219989337680E+27,   1.339555198294029E+29,   2.277243837099849E+30, 
         4.099038906779728E+31,   7.788173922881483E+32,   1.557634784576297E+34},
     {   1.216451004088320E+17,   1.216451004088320E+17,   2.432902008176640E+17,   7.298706024529920E+17,   2.919482409811968E+18,   1.459741204905984E+19, 
         8.758447229435904E+19,   6.130913060605133E+20,   4.904730448484106E+21,   4.414257403635696E+22,   4.414257403635696E+23,   4.855683143999265E+24, 
         5.826819772799119E+25,   7.574865704638854E+26,   1.060481198649439E+28,   1.590721797974159E+29,   2.545154876758655E+30,   4.326763290489713E+31, 
         7.788173922881483E+32,   1.479753045347482E+34,   2.959506090694964E+35},
     {   2.432902008176640E+18,   2.432902008176640E+18,   4.865804016353280E+18,   1.459741204905984E+19,   5.838964819623936E+19,   2.919482409811968E+20, 
         1.751689445887181E+21,   1.226182612121027E+22,   9.809460896968212E+22,   8.828514807271392E+23,   8.828514807271392E+24,   9.711366287998532E+25, 
         1.165363954559824E+27,   1.514973140927771E+28,   2.120962397298879E+29,   3.181443595948319E+30,   5.090309753517310E+31,   8.653526580979427E+32, 
         1.557634784576297E+34,   2.959506090694964E+35,   5.919012181389928E+36}
   };

#else
   static unsigned short int is_initialized = 0;
   static double **factorial_products;

   /* factorial_array[n] = n! */
   static double factorial_array[]={1.,1.,2.,6.,24.,120.,720.,5040.,40320.,362880.,
      3628800.,39916800.,479001600.,6227020800.,87178291200.,1307674368000.,20922789888000.,
      355687428096000.,6402373705728000.,121645100408832000.,
      2.43290200817664e+18,5.109094217170944e+19};

   /* If this routine hasn't been called yet, allocate the factorial products,
    * factorial_products[i][j] = i!j!.  Note that we never de-allocate this
    * memory, it is freed only when the program exits.  This is no big loss
    * since we only allocate it once and even then it's only a few kbytes.
    */
   if (!is_initialized) {
      is_initialized = 1;
      factorial_products = dmatrix(0,MAX_FACTORIAL,0,MAX_FACTORIAL);
      for(i=0;i<=MAX_FACTORIAL;i++)
         for(j=0;j<=MAX_FACTORIAL;j++)
            factorial_products[i][j] = factorial_array[i] * factorial_array[j];
   }
#endif

#ifdef N_CHECKVAL
   /* Error indicators */
   if (M<0 || N<1 || M+N>21) {
      fprintf(stderr,"Error: Bad order for interp_coefs_1: M=%d, N=%d\n",M,N);
      exit(1);
   }
#endif

   /* First get rid of the cases where x is close to 0 or to 1;
    * we need this branching because for the general case we divide
    * by x and (x-1).
    */
   if (fabs(x)<INTERP_XMIN) {
      for(i= -M;i<=N;i++) c[i]=0.;
      c[0] = 1.;
      return;
   }
   if (fabs(x-1.)<INTERP_XMIN) {
      for(i= -M;i<=N;i++) c[i]=0.;
      c[1] = 1.;
      return;
   }

   /* Compute product = (x+M)(x+M-1) ... (x+1)x(x-1) ... (x-N) */
   product = 1.;
   for(i= -M;i<=N;i++) product *= x-i;

   /* Compute interpolation coefficients */
   for(i=N;i>= -M;i--) {
      if ((N-i)%2) {
         c[i] = -product / ( (x-i) * factorial_products[N-i][M+i] );
      } else {
         c[i] =  product / ( (x-i) * factorial_products[N-i][M+i] );
      }
   }
}

/* forward_grid_interp_1
 * *** INTERPOLATES A VALUE FROM A GRID OF POINTS ***
 *
 * We are given some function on a grid, func(x,y) for integer
 * x and y, and we have the coordinates x[0..length-1], y[0..length-1].
 * This function will interpolate to find f[0..length-1], where:
 * f[i] = func(x[i],y[i]).
 *
 * A 2D polynomial fit is used on the grid of points with dimension
 * 2*M+1 by 2*M+1.
 *
 * The grid MUST be defined with xrange [floor(x_min)-M .. ceil(x_max)+M]
 * and similarly for y (this is NOT checked for!)
 *
 * Note that we do not clear the output before adding the result, thus
 * you must initialize the output before calling this routine.
 */
void forward_grid_interp_1(double **func, double *x, double *y, double *f,
   int M, long length) {

   long k;
   int i,j;
   double xfrac, yfrac;
   long xint, yint;
   int N = M+1; /* the interpolating coefficients go from -M to N */
   double *c, *intermed, *funcptr;

   /* allocate memory */
   c = dvector(-M,N);
   intermed = dvector(-M,N);

   for(k=0;k<length;k++) { /* loop over the length points to be interpolated */

      /* Split x and y into their integer and fractional parts */
      xfrac = *x - (xint = (long) floor(*x));
      yfrac = *y - (yint = (long) floor(*y));

      /* Do vertical interpolation to get intermed[i]=func[xint+i][*y] */
      interp_coefs_1(yfrac,M,N,c);
      for(i= -M;i<=N;i++) {
         intermed[i] = 0.;
         funcptr = func[xint+i] + yint;
         for(j= -M;j<=N;j++) intermed[i] += c[j] * funcptr[j];
      }

      /* Now do the horizontal interpolation */
      interp_coefs_1(xfrac,M,N,c);
      for(i= -M;i<=N;i++) { *f += c[i] * intermed[i]; }

      /* Increment pointers to the coordinates and output */
      x++; y++; f++;

   } /* end k loop */

   /* de-allocate memory */
   free_dvector(c,-M,N);
   free_dvector(intermed,-M,N);

}

/* reverse_grid_interp_1
 * *** TRANSPOSE OF INTERPOLATION OF A VALUE FROM A GRID OF POINTS ***
 *
 * This is a transpose routine for forward_grid_interp_1, whose description
 * follows: (Note that this routine, like forward_grid_interp_1, DOES NOT
 * clear the output before adding the result!)
 *
 * ================
 *
 * We are given some function on a grid, func(x,y) for integer
 * x and y, and we have the coordinates x[0..length-1], y[0..length-1].
 * This function will interpolate to find f[0..length-1], where:
 * f[i] = func(x[i],y[i]).
 *
 * A 2D polynomial fit is used on the grid of points with dimension
 * 2*M+1 by 2*M+1.
 *
 * The grid MUST be defined with xrange [floor(x_min)-M .. ceil(x_max)+M]
 * and similarly for y (this is NOT checked for!)
 *
 * Note that we do not clear the output before adding the result, thus
 * you must initialize the output before calling this routine.
 */
void reverse_grid_interp_1(double **func, double *x, double *y, double *f,
   int M, long length) {

   long k;
   int i,j;
   double xfrac, yfrac;
   long xint, yint;
   int N = M+1; /* the interpolating coefficients go from -M to N */
   double *c, *intermed, *funcptr;

   /* allocate memory */
   c = dvector(-M,N);
   intermed = dvector(-M,N);

   for(k=0;k<length;k++) { /* loop over the length points to be interpolated */

      /* Split x and y into their integer and fractional parts */
      xfrac = *x - (xint = (long) floor(*x));
      yfrac = *y - (yint = (long) floor(*y));

      /* Do the horizontal interpolation to get intermed[i]=func[xint+i][*y] */
      interp_coefs_1(xfrac,M,N,c);
      for(i= -M;i<=N;i++) { intermed[i] = c[i] * (*f); }

      /* Now do the vertical interpolation */
      interp_coefs_1(yfrac,M,N,c);
      for(i= -M;i<=N;i++) {
         funcptr = func[xint+i] + yint;
         for(j= -M;j<=N;j++) funcptr[j] += c[j] * intermed[i];
      }

      /* Increment pointers to the coordinates and output */
      x++; y++; f++;

   } /* end k loop */

   /* de-allocate memory */
   free_dvector(c,-M,N);
   free_dvector(intermed,-M,N);

}

/*
 * Contains codes for the following operations on the sphere:
 *
 * optimum_length_1: determined the optimal length of FFT to use
 *    when it is expensive to evaluate the function of which we want
 *    the Fourier transform
 * latitude_syn_1: SHT synthesis, latitude direction
 * sht_grid_synthesis_1: SHT synthesis onto equirectangular grid
 * latitude_syn_vector_1: spin 1 vector SHT synthesis, latitude direction
 * sht_grid_synthesis_vector_1: spin 1 vector SHT synthesis onto
 *     equirectangular grid
 * latitude_syn_polar_1: spin 2 tensor SHT synthesis, latitude direction
 * sht_grid_synthesis_polar_1: spin 2 tensor SHT synthesis onto
 *     equirectangular grid
 * latitude_ana_1: SHT analysis, latitude direction
 * sht_grid_analysis_1: SHT analysis on equirectangular grid
 * latitude_ana_vector_1: spin 1 vector SHT analysis, latitude direction
 * sht_grid_analysis_vector_1: spin 1 vector SHT analysis on
 *     equirectangular grid
 * latitude_ana_polar_1: spin 2 tensor SHT analysis, latitude direction
 * sht_grid_analysis_polar_1: spin 2 tensor SHT analysis on
 *     equirectangular grid
 * sphere_step_1: computes geometry of terminal point of great circle
 *     traverse on a sphere
 * sphere_coord_bound: puts spherical coordinates in standard range
 *     on sphere
 * forward_sphere_pm_1: mesh=>particle interpolation on sphere
 * reverse_sphere_pm_1: particle=>mesh interpolation on sphere
 * pixel_synthesis_1: SHT synthesis to arbitrary pixelization
 * pixel_analysis_1: SHT analysis from arbitrary pixelization
 * pixel_convolution_1: spherical convolution on arbitrary pixelization
 * pixel_convolution_2: spherical convolution on arbitrary pixelization
 *     using SPHERE_PIXEL structure
 */

/* optimum_length_1
 * *** DETERMINES THE BEST LENGTH OF FOURIER TRANSFORM TO USE ***
 *
 * Returns the value of length that minimizes:
 *
 * t = length * ( n + beta * mantissa )
 *
 * where mantissa is the largest odd number dividing length.  This
 * equation is the large-n limit of the problem of finding the optimal
 * Fourier transform length to use to determine ftilde_m when computing
 * f(x) is computationally expensive, i.e. >> log n steps, and when we
 * need at least n points (due to band-limited f).  It is used
 * to gain nearly a factor of two in computation speed of spherical
 * harmonic transforms.  Note that we require the output to be a
 * multiple of 4 so that the appropriate symmetry behavior is
 * acquired in each quadrant.
 */

unsigned long optimum_length_1(unsigned long n, float beta) {

   unsigned long nn2 = 1;
   unsigned long length, length_old, mantissa;
   unsigned long n1;
   float t, t_old;

   /* If the input is less than or equal to 4, we want a length 4 FFT
    * because the length must be a multiple of 4.
    */
   if (n<=4) return 4;

   /* Find nn2, the largest power of 2 not exceeding n. */
   while (nn2<n) nn2 <<= 1;

   /* If n is a power of 2, go home. */
   if (nn2==n) return n;

   /* If n is one less than a power of 2, also go home.  Note that
    * we need to do this to avoid a crash later in the code!!!
    */
   if (nn2==n+1) return nn2;

   /* Otherwise, we want to start with nn*2 and work downward
    * through numbers of the form mantissa * power of 2 that are
    * greater than n until the computation time begins to rise.
    */
   length = nn2;
   t = (float)length * ((float)n + beta);
   t_old = t*2.0+1.0;
   n1 = n-1;

   length_old = length; /* this is re-initialized before it is used */
                        /* but this avoids the compiler warning.    */

   while (t<t_old) {
      length_old = length;
      do {nn2 >>= 1;} while (nn2&n); /* This line finds the next power
                                      * of 2 for nn2 that yields a different
                                      * length.  It will crash if there are
                                      * no zeroes in the binary expansion of
                                      * n, i.e. for Mersenne numbers.  The
                                      * code at the beginning of this routine
                                      * protects against this error.
                                      */
      if (nn2<4) {
         t = t_old + 1.0; /* breaks loop */
      } else {
         length = ( mantissa = n1/nn2 + 1 ) *nn2;
         t_old = t;
         t = length * ( n + beta * mantissa );
      }
   }

   return length_old;
}

/* latitude_syn_1
 * *** TRANSFORMS FROM SPHERICAL HARMONIC A_LM TO A_M(THETA) ***
 *
 * This function performs the latitude part of a spherical harmonic transform; the function
 * is also capable of doing a convolution on the a_lm's before doing the spherical
 * harmonic transform.
 *
 * The storage conventions are:
 *
 * ALM = coefficients according to SPHERE_MODES->coefs
 * AMTheta[M][y] = Re a_M(pi/2 - y*dtheta) [M>=0]
 *                 Im a_|M|(pi/2 - y*dtheta) [M<0]
 *
 * func(theta,phi) = Re sum_{M>=0} a_M(theta) e^(iM phi)
 *
 * Dimensions: AMTheta[-lmax..lmax][-ntheta..ntheta]
 * (Note: we use this range but if ALM is defined outside of this,
 * this will not cause a problem.)
 *
 * If convolve_flag!=0, ALM[L][M] is multiplied by convolve_kernel[L] prior
 * to the transform.
 */

void latitude_syn_1(double **ALM, double **AMTheta, double dtheta, long ntheta, long lmax,
   double *convolve_kernel, unsigned short int convolve_flag) {

   long L,M,y,j;
   double *RecA, *RecB; /* recursion coefficients for generating Y_LM */
   double *RecAp, *RecBp;
   double *Cmodes, *Smodes;
   double *Cmp, *Smp;
   double costheta, sintheta;
   double *sinMGtheta;
   int *lambda_exp;
   int lambda_exp_current;
   double InitializerCoef;
   double AbsYLp1M,AbsYLM,AbsYLm1M;
   double AMCEven, AMCOdd, AMSEven, AMSOdd;
   unsigned short int latitude_parity;

   /* Allocate auxiliary arrays */
   RecA = dvector(0,lmax);
   RecB = dvector(0,lmax);
   Cmodes = dvector(0,lmax);
   Smodes = dvector(0,lmax);
   sinMGtheta = dvector(0,ntheta);
   lambda_exp = ivector(0,ntheta);

   /* Initialize sin^(M-G)(theta) array and exponent overflow array */
   for(y=0;y<=ntheta;y++) {
      sinMGtheta[y] = 1.;
      lambda_exp[y] = 0;
   }

   /* Outer loop over the M-values */
   for(M=0;M<=lmax;M++) {

      /* Initialize recurrance relation coefficients */
      for(L=M;L<=lmax;L++) {
         RecA[L] = sqrt((2*L+3.0)/((L+1)*(L+1)-M*M));
         RecB[L] = sqrt((L*L-M*M)/(2*L-1.0))*RecA[L];
         RecA[L] *= sqrt(2*L+1.0);
      }
      InitializerCoef=OneOverTwoPi;
      for(j=1;j<=M;j++) InitializerCoef *= 1.0 + 0.5/j;
      InitializerCoef = sqrt(InitializerCoef);
      if (M%2==1) InitializerCoef *= -1;

      /* and construct the vectors of cosinelike and sinelike coefficients */
      for(L=M;L<=lmax;L++) {
         Cmodes[L] = ALM[L][M];
         Smodes[L] = ALM[L][-M];
      }
      if (M==0) for(L=0;L<=lmax;L++) {
         Smodes[L] = 0; /* no sin(0phi) mode */
         Cmodes[L] /= SqrtTwo;
      }

      /* do the pre-transform convolution, if necessary */
      if (convolve_flag) {
         for(L=M;L<=lmax;L++) {
            Cmodes[L] *= convolve_kernel[L];
            Smodes[L] *= convolve_kernel[L];
         }
      }

      /* Now run through all the points */
      for(y=0;y<=ntheta;y++) {
         /* Note that y*dtheta is the latitude but theta is the colatitude;
          * this is why sin and cos look backward but yes, this is correct.
          */
         costheta = sin(y*dtheta);
         sintheta = cos(y*dtheta);

         /* Initialize the spherical harmonics at L=|M|; we will   
          * use the recurrance relation, based on NR 6.8.2,7 to    
          * find the normalized latitude part of the spherical     
          * harmonic.  The recurrance is:                          
          * sqrt((L^2-M^2)/(2L+1) * |Y_L_M| =
          *                        sqrt(2L-1) * cos(theta)|Y_L-1_M|
          *     - sqrt(((L-1)^2-M^2)/(2L-3)) * |Y_L-2_M|           
          * AbsYLMs are normalized to Int(AbsYLM sin^G theta)^2 dcostheta = 1/Pi
          */
         if (sinMGtheta[y] < LAMBDA_MIN) {
            sinMGtheta[y] *= INV_LAMBDA_MIN;
            lambda_exp[y]++;
         }
         lambda_exp_current = lambda_exp[y];
         AbsYLm1M = 0;
         AbsYLM = InitializerCoef * sinMGtheta[y];

         /* Increment sinMGtheta for next use */
         sinMGtheta[y] *= sintheta;

         /* Initialize pointers */
         RecAp = RecA + M;
         RecBp = RecB + M;
         Cmp = Cmodes + M;
         Smp = Smodes + M;

         /* Initialize the opposite-parity arrays and parity counter */
         AMCEven = AMCOdd = AMSEven = AMSOdd = 0.;
         latitude_parity = 0;

         /* Now go through the values of L */
         for(L=M;L<=lmax;L++) {

            if (lambda_exp_current>0) {
               /* If lambda_exp_current>0 we don't need to do the
                * multiplication/addition of the output, but we must
                * see whether Ylm is large enough that we can reduce
                * the number of orders of underflow, lambda_exp_current.
                */
               if (fabs(AbsYLM) > 1.0) {
                  lambda_exp_current--;
                  AbsYLM *= LAMBDA_MIN;
                  AbsYLm1M *= LAMBDA_MIN;
               }

            }

            if (lambda_exp_current==0) {
               /* Do the multiplication for each latitude ring
                * Note that we only increment the even or odd parity
                * a_m(theta)'s depending on the parity of the
                * particular Ylm.
                */
               if (latitude_parity) {
                  AMCOdd  += AbsYLM * (*Cmp);
                  AMSOdd  += AbsYLM * (*Smp);
               } else {
                  AMCEven += AbsYLM * (*Cmp);
                  AMSEven += AbsYLM * (*Smp);
               }

            }

            /* and then increment the Ylm and pointers */
            AbsYLp1M = (*RecAp) * costheta * AbsYLM - (*RecBp) * AbsYLm1M;
            RecAp++; RecBp++;
            Cmp++; Smp++;
            AbsYLm1M = AbsYLM;
            AbsYLM = AbsYLp1M;

            latitude_parity ^= 1; /* switches between 0 and 1. */

         } /* end L loop */

         /* Now put these results into the output arrays */
         AMTheta[ M][ y] = AMCEven + AMCOdd;
         AMTheta[ M][-y] = AMCEven - AMCOdd;
         if (M!=0) {
            AMTheta[-M][ y] = AMSEven + AMSOdd;
            AMTheta[-M][-y] = AMSEven - AMSOdd;
         }

      } /* end y loop */

   } /* end M loop */

   /* De-allocate auxiliary arrays */
   free_dvector(RecA,0,lmax);
   free_dvector(RecB,0,lmax);
   free_dvector(Cmodes,0,lmax);
   free_dvector(Smodes,0,lmax);
   free_dvector(sinMGtheta,0,ntheta);
   free_ivector(lambda_exp,0,ntheta);
}

/* sht_grid_synthesis_1
 * *** TRANSFORMS FROM SPHERICAL A_LM TO FUNC(THETA,PHI) ON EQUIRECTANGULAR GRID ***
 *
 * The a_lm must be in the SPHERE_MODES->coefs convention
 * The grid output is:
 *
 * func_map[x][y] += f(phi = x*delta, theta = y*delta)
 *
 * where delta = 2pi/nspace
 * and the output range of func_map is [xmin..xmax][ymin..ymax].
 * (Note: we use this range but if func_map is defined outside of this,
 * this will not cause a problem.)
 *
 * nspace MUST be a multiple of 4.  It is best for it to be a power of two, or a
 * power of two times a small odd number (this is for the FFT).  If nspace is 
 * a multiple of 4 with lots of odd factors, this routine will still work but will
 * be slow.
 *
 * If convolve_flag!=0, the function is multiplied by the convolve_kernel first, then
 * transformed.
 */

void sht_grid_synthesis_1(double **ALM, double **func_map, long lmax, long nspace,
   long xmin, long xmax, long ymin, long ymax, double *convolve_kernel,
   unsigned short int convolve_flag) {

   long nspace_coarse, nspace_max, quarter_nspace_max, two_nspace_max;
   double dtheta;
   double **FuncThetaPhi;
   double **AMTheta;
   double **AMFine;
   double *AMFinePtrM, *AMFinePtrP;
   double *AMThetaPtrM, *AMThetaPtrP;
   double *FourierStackFine, *FourierStackCoarse;
   long two_nspace, half_nspace, quarter_nspace;
   long ntheta_m, two_ntheta_m, four_ntheta_m, eight_ntheta_m;
   long M,i,x,y,j;
   long xoffset, yoffset;

   /* Check that we were given a multiple of 4 */
   if (nspace % 4) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld should be a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Compute useful quantities related to nspace */
   two_nspace = nspace << 1;
   half_nspace = nspace >> 1;
   quarter_nspace = nspace >> 2;

   /* Determine how finely we need to sample theta.  Here dtheta is the spacing of our points in theta and
    * ntheta is the total number spaced around the entire great circle meridian (2pi).
    * We also determine nspace_max and quarter_nspace_max; nspace_max will be the maximum
    * number of points we need to consider on the entire meridian, quarter_nspace_max will
    * be only for the quarter-meridian from 0 to pi/2.
    */
   nspace_max = 0;
   nspace_coarse = optimum_length_1( 2*lmax+1, BETA_FFT_SHT);
      /* The "+1" in the optimum_length_1 argument is there so that we do not have
       * any power at the Nyquist frequency itself (this would result in aliasing
       * and consequent loss of information).
       */
   ntheta_m = nspace_coarse >> 2;
   two_ntheta_m = ntheta_m << 1;
   four_ntheta_m = ntheta_m << 2;
   eight_ntheta_m = ntheta_m << 3;
   dtheta = TwoPi/nspace_coarse;
   nspace_max = nspace_coarse;
   quarter_nspace_max = nspace_max >> 2;
   two_nspace_max = nspace_max << 1;

   /* Check that we are over-sampling */
   if (nspace_coarse > nspace) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld, lmax=%ld, this is not oversampled.\n",nspace,lmax);
      exit(1);
   }

   /* Allocate and compute the a_M(theta)'s */
   AMTheta = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   AMFine = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   FourierStackFine = dvector(0,two_nspace-1);
   FourierStackCoarse = dvector(0,two_nspace_max-1);
   latitude_syn_1(ALM,AMTheta,dtheta,ntheta_m,lmax,convolve_kernel,convolve_flag);

   /* For each M, we do an exact trigonometric interpolation to yield
    * the values of am(theta) at a grid of points spaced in theta by
    * TwoPi/nspace.  The method is to consider a complete meridian
    * starting at the S pole, going up to the N pole, and then back
    * down at the opposite (+180deg) longitude.
    */
   for(M=0;M<=lmax;M++) {

      /* Set up the arrays for the latitude FFT */
      AMThetaPtrP = AMTheta[ M] - ntheta_m;
      AMThetaPtrM = AMTheta[-M] - ntheta_m;
      for(i=0;i<two_nspace_max;i++) FourierStackCoarse[i] = 0;
      FourierStackCoarse[0] = AMThetaPtrP[0] / four_ntheta_m;
      FourierStackCoarse[1] = AMThetaPtrM[0] / four_ntheta_m;

      /* the symmetry across theta => -theta is different for odd and even M.  The division by
       * four_ntheta_m is designed so that the forward and backward Fourier transforms will be
       * inverses of each other, without any multiplying factors.
       */
      if (M%2==0) { /* M even */
         for(i=1;i<=two_ntheta_m;i++) {
            FourierStackCoarse[eight_ntheta_m - 2*i   ] = (FourierStackCoarse[2*i  ] = AMThetaPtrP[i]/four_ntheta_m);
            FourierStackCoarse[eight_ntheta_m - 2*i +1] = (FourierStackCoarse[2*i+1] = AMThetaPtrM[i]/four_ntheta_m);
         }
      } else { /* M odd */
         for(i=1;i<=two_ntheta_m;i++) {
            FourierStackCoarse[eight_ntheta_m - 2*i   ] = -( FourierStackCoarse[2*i  ] = AMThetaPtrP[i]/four_ntheta_m);
            FourierStackCoarse[eight_ntheta_m - 2*i +1] = -( FourierStackCoarse[2*i+1] = AMThetaPtrM[i]/four_ntheta_m);
            /* We have set FourierStack[two_ntheta_m], FourierStack[two_ntheta_m+1] equal to
             * negative of their actual values.  This doesn't matter since they are zero anyway.
             */
         }
      }

      /* Now do the FFT to get the Fourier transform of a_M(theta) */
      fourier_trans_2(FourierStackCoarse-1,four_ntheta_m,-1);

      /* We will use FourierStackFine to do the forward FFT to get a_M(theta) at the new
       * points.  Begin by clearing it, then copying over positive, then negative frequency
       * modes, finally do the FFT itself.
       */
      for(i=0;i<two_nspace;i++) FourierStackFine[i] = 0;
      for(i=0;i<four_ntheta_m;i++) FourierStackFine[i] = FourierStackCoarse[i];
      for(i=1;i<=four_ntheta_m;i++) FourierStackFine[two_nspace - i] = FourierStackCoarse[eight_ntheta_m - i];
      fourier_trans_2(FourierStackFine-1,nspace,1);

      /* Put the interpolated values into AMFine */
      AMFinePtrP = AMFine[ M] - quarter_nspace;
      AMFinePtrM = AMFine[-M] - quarter_nspace;
      for(i=0;i<=half_nspace;i++) {
         AMFinePtrP[i] = FourierStackFine[2*i  ];
         AMFinePtrM[i] = FourierStackFine[2*i+1];
      }

   } /* end M loop */

   /* Reallocate some memory */
   free_dmatrix(AMTheta,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   FuncThetaPhi = dmatrix(0,nspace-1,-quarter_nspace,quarter_nspace);

   /* For each j, we do the longitude FFT */
   for(j=-quarter_nspace;j<=quarter_nspace;j++) {

      /* load elements of AMFine into FourierStack */
      for(i=0;i<two_nspace;i++) FourierStackFine[i]=0;
      FourierStackFine[0] = AMFine[0][j];
      for(M=1;M<=lmax;M++) {
         FourierStackFine[2*M] = AMFine[M][j];
         FourierStackFine[2*M+1]= AMFine[-M][j];
      }

      /* FFT from AM(theta) to A(phi)(theta) */
      fourier_trans_2(FourierStackFine-1,nspace,1);
      for(i=0;i<nspace;i++) FuncThetaPhi[i][j] = FourierStackFine[2*i];

   } /* end j loop */

   /* Now build the output map */
   xoffset = (xmin>=0)? 0 : nspace * ((-xmin)/nspace + 1);
   yoffset = (ymin>=0)? 0 : nspace * ((-ymin)/nspace + 1);
   for(x=xmin;x<=xmax;x++) {
      for(y=ymin;y<=ymax;y++) {
         i = (x + xoffset) % nspace;
         j = (y + yoffset) % nspace;
         if (j>=3*quarter_nspace) j -= nspace;
         if (j>quarter_nspace) {
            i = (i + half_nspace) % nspace;
            j = half_nspace - j;
         }

         func_map[x][y] += FuncThetaPhi[i][j];
      }
   }

   /* Clean up memory */
   free_dmatrix(AMFine,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dvector(FourierStackFine,0,two_nspace-1);
   free_dvector(FourierStackCoarse,0,two_nspace_max-1);
   free_dmatrix(FuncThetaPhi,0,nspace-1,-quarter_nspace,quarter_nspace);

}

/* latitude_syn_vector_1
 * *** TRANSFORMS FROM VECTOR SPHERICAL HARMONIC A_LM TO A_M(THETA) ***
 *
 * This function performs the latitude part of a spherical harmonic transform.
 * The function is also capable of doing a convolution on the a_lm's before
 * doing the spherical harmonic transform.
 *
 * The storage conventions are:
 *
 * ALM_V, ALM_A = coefficients according to SPHERE_MODES->coefs
 *
 * (Note: "V" and "A" represent the vector and axial-vector spherical
 * harmonics.)
 *
 * AMTheta[M][y] = Re a_M(pi/2 - y*dtheta) [M>=0]
 *                   Im a_|M|(pi/2 - y*dtheta) [M<0]
 *
 * func(theta,phi) = Re sum_{M>=0} a_M(theta) e^(iM phi)
 *
 * Dimensions: AMTheta_{v1,v2}[-lmax..lmax][-ntheta..ntheta]
 * (Note: we use this range but if ALM is defined outside of this,
 * this will not cause a problem.)
 *
 * If convolve_flag!=0, ALM_V[L][M] is multiplied by convolve_kernel_V[L] prior
 * to the transform, and similarly for ALM_A and convolve_kernel_A.
 */

void latitude_syn_vector_1(double **ALM_V, double **ALM_A, double **AMTheta_1,
   double **AMTheta_2, double dtheta, long ntheta, long lmax,
   double *convolve_kernel_V, double *convolve_kernel_A,
   unsigned short int convolve_flag) {

   long L,M,y,j;
   double *RecA, *RecB; /* recursion coefficients for generating Y_LM */
   double *RecAp, *RecBp;
   double *RecC;
   double *RecCp;
   double *Cmodes_V, *Smodes_V, *Cmodes_A, *Smodes_A;
   double *Cmp_V, *Smp_V, *Cmp_A, *Smp_A;
   double costheta, sintheta, Msintheta;
   double AbsWLM, AbsXLM;
   double *sinMGtheta;
   int *lambda_exp;
   int lambda_exp_current;
   double InitializerCoef;
   double normalization;
   double AbsYLp1M,AbsYLM,AbsYLm1M;
   double AMCEven_1, AMCOdd_1, AMSEven_1, AMSOdd_1;
   double AMCEven_2, AMCOdd_2, AMSEven_2, AMSOdd_2;
   unsigned short int latitude_parity;

   /* Allocate auxiliary arrays */
   RecA = dvector(0,lmax);
   RecB = dvector(0,lmax);
   RecC = dvector(0,lmax);
   Cmodes_V = dvector(0,lmax);
   Smodes_V = dvector(0,lmax);
   Cmodes_A = dvector(0,lmax);
   Smodes_A = dvector(0,lmax);
   sinMGtheta = dvector(0,ntheta);
   lambda_exp = ivector(0,ntheta);

   /* Initialize sin^(M-G)(theta) array and exponent overflow array */
   for(y=0;y<=ntheta;y++) {
      sinMGtheta[y] = 1.;
      lambda_exp[y] = 0;
   }

   /* Outer loop over the M-values */
   for(M=0;M<=lmax;M++) {

      /* Initialize recurrance relation coefficients */
      for(L=M;L<=lmax;L++) {

         /* Y recursions */
         RecA[L] = sqrt((2*L+3.0)/((L+1)*(L+1)-M*M));
         RecB[L] = sqrt((L*L-M*M)/(2*L-1.0))*RecA[L];
         RecA[L] *= sqrt(2*L+1.0);

         /* X' recursions -- these depend on Y. */
         RecC[L] = (L>0)? sqrt( (2*L+1.0)/(2*L-1.0) * (L*L-M*M) ) : 0;
      }
      InitializerCoef=OneOverTwoPi;
      for(j=1;j<=M;j++) InitializerCoef *= 1.0 + 0.5/j;
      InitializerCoef = sqrt(InitializerCoef);
      if (M%2==1) InitializerCoef *= -1;

      /* and construct the vectors of cosinelike and sinelike coefficients */
      for(L=M;L<=lmax;L++) {
         Cmodes_V[L] = ALM_V[L][M];
         Smodes_V[L] = ALM_V[L][-M];
         Cmodes_A[L] = ALM_A[L][M];
         Smodes_A[L] = ALM_A[L][-M];
      }
      if (M==0) for(L=0;L<=lmax;L++) {
         Smodes_V[L] = Smodes_A[L] = 0; /* no sin(0phi) mode */
         Cmodes_V[L] /= SqrtTwo;
         Cmodes_A[L] /= SqrtTwo;
      }

      /* do the pre-transform convolution, if necessary */
      if (convolve_flag) {
         for(L=M;L<=lmax;L++) {
            Cmodes_V[L] *= convolve_kernel_V[L];
            Smodes_V[L] *= convolve_kernel_V[L];
            Cmodes_A[L] *= convolve_kernel_A[L];
            Smodes_A[L] *= convolve_kernel_A[L];
         }
      }

      /* and now apply the normalization factor, i.e. the conversion
       * between (W'+iX')_lm and (v1+iv2).
       */
      for(L=M;L<=lmax;L++) {
         normalization = (L>=1)? 1./sqrt((double)L*(L+1)) :0 ;
         Cmodes_V[L] *= normalization;
         Smodes_V[L] *= normalization;
         Cmodes_A[L] *= normalization;
         Smodes_A[L] *= normalization;
      }

      /* Now run through all the points */
      for(y=0;y<=ntheta;y++) {
         /* Note that y*dtheta is the latitude but theta is the colatitude;
          * this is why sin and cos look backward but yes, this is correct.
          */
         costheta = sin(y*dtheta);
         sintheta = cos(y*dtheta);
         Msintheta = M * sintheta;

         /* Initialize the spherical harmonics at L=|M|; we will   
          * use the recurrance relation, based on NR 6.8.2,7 to    
          * find the normalized latitude part of the spherical     
          * harmonic.  The recurrance is:                          
          * sqrt((L^2-M^2)/(2L+1) * |Y_L_M| =
          *                        sqrt(2L-1) * cos(theta)|Y_L-1_M|
          *     - sqrt(((L-1)^2-M^2)/(2L-3)) * |Y_L-2_M|           
          * AbsYLMs are normalized to Int(AbsYLM sin^G theta)^2 dcostheta = 1/Pi
          */
         if (sinMGtheta[y] < LAMBDA_MIN) {
            sinMGtheta[y] *= INV_LAMBDA_MIN;
            lambda_exp[y]++;
         }
         lambda_exp_current = lambda_exp[y];
         AbsYLm1M = 0;
         AbsYLM = InitializerCoef * sinMGtheta[y];

         /* Increment sinMGtheta for next use */
         if (M>=1) sinMGtheta[y] *= sintheta;

         /* Initialize pointers */
         RecAp = RecA + M;
         RecBp = RecB + M;
         RecCp = RecC + M;
         Cmp_V = Cmodes_V + M;
         Smp_V = Smodes_V + M;
         Cmp_A = Cmodes_A + M;
         Smp_A = Smodes_A + M;

         /* Initialize the opposite-parity arrays and parity counter */
         AMCEven_1 = AMCOdd_1 = AMSEven_1 = AMSOdd_1 = 0.;
         AMCEven_2 = AMCOdd_2 = AMSEven_2 = AMSOdd_2 = 0.;
         latitude_parity = 0;

         /* Now go through the values of L */
         for(L=M;L<=lmax;L++) {

            if (lambda_exp_current>0) {
               /* If lambda_exp_current>0 we don't need to do the
                * multiplication/addition of the output, but we must
                * see whether Ylm is large enough that we can reduce
                * the number of orders of underflow, lambda_exp_current.
                */
               if (fabs(AbsYLM) > 1.0) {
                  lambda_exp_current--;
                  AbsYLM *= LAMBDA_MIN;
                  AbsYLm1M *= LAMBDA_MIN;
               }

            }

            if (lambda_exp_current==0) {

               /* First we need to compute W' and X'.
                * Note that these actually compute sin(theta) times the gradient
                * of Ylm, we re-insert the factor of 1/sin(theta) later.
                */
               AbsWLM = M * AbsYLM;
               AbsXLM = -L * costheta * AbsYLM + (*RecCp) * AbsYLm1M;

               /* Do the multiplication for each latitude ring
                * Note that we only increment the even or odd parity
                * a_m(theta)'s depending on the parity of the
                * particular Ylm.
                */
               if (latitude_parity) {
                  AMCOdd_1   -= AbsWLM * (*Smp_V);
                  AMSOdd_1   += AbsWLM * (*Cmp_V);
                  AMCEven_2  += AbsXLM * (*Cmp_V);
                  AMSEven_2  += AbsXLM * (*Smp_V);
                  AMCOdd_2   -= AbsWLM * (*Smp_A);
                  AMSOdd_2   += AbsWLM * (*Cmp_A);
                  AMCEven_1  -= AbsXLM * (*Cmp_A);
                  AMSEven_1  -= AbsXLM * (*Smp_A);
               } else {
                  AMCEven_1  -= AbsWLM * (*Smp_V);
                  AMSEven_1  += AbsWLM * (*Cmp_V);
                  AMCOdd_2   += AbsXLM * (*Cmp_V);
                  AMSOdd_2   += AbsXLM * (*Smp_V);
                  AMCEven_2  -= AbsWLM * (*Smp_A);
                  AMSEven_2  += AbsWLM * (*Cmp_A);
                  AMCOdd_1   -= AbsXLM * (*Cmp_A);
                  AMSOdd_1   -= AbsXLM * (*Smp_A);
               }
            }

            /* and then increment the Ylm and pointers */
            AbsYLp1M = (*RecAp) * costheta * AbsYLM - (*RecBp) * AbsYLm1M;
            RecAp++; RecBp++; RecCp++;
            Cmp_V++; Smp_V++; Cmp_A++; Smp_A++;
            AbsYLm1M = AbsYLM;
            AbsYLM = AbsYLp1M;

            latitude_parity ^= 1; /* switches between 0 and 1. */

         } /* end L loop */

         /* This block of code will add our results to the
          * output arrays.  This step is unnecessary (in fact, the code
          * will crash!) if M<1 and sintheta==0, hence we block that case.
          */
         if (M==1 || fabs(sintheta)>MIN_SIN_THETA) {

            /* Divide the resulting Q and U by sin(theta) if necessary */
            if (M==0) {
               AMCEven_1 /= sintheta;
               AMCEven_2 /= sintheta;
               AMCOdd_1  /= sintheta;
               AMCOdd_2  /= sintheta;
            }

            /* Now put these results into the output arrays */
            AMTheta_1[ M][ y] = AMCEven_1 + AMCOdd_1;
            AMTheta_1[ M][-y] = AMCEven_1 - AMCOdd_1;
            AMTheta_2[ M][ y] = AMCEven_2 + AMCOdd_2;
            AMTheta_2[ M][-y] = AMCEven_2 - AMCOdd_2;
            if (M!=0) {
               AMTheta_1[-M][ y] = AMSEven_1 + AMSOdd_1;
               AMTheta_1[-M][-y] = AMSEven_1 - AMSOdd_1;
               AMTheta_2[-M][ y] = AMSEven_2 + AMSOdd_2;
               AMTheta_2[-M][-y] = AMSEven_2 - AMSOdd_2;
            }
         }

      } /* end y loop */

   } /* end M loop */

   /* De-allocate auxiliary arrays */
   free_dvector(RecA,0,lmax);
   free_dvector(RecB,0,lmax);
   free_dvector(RecC,0,lmax);
   free_dvector(Cmodes_V,0,lmax);
   free_dvector(Smodes_V,0,lmax);
   free_dvector(Cmodes_A,0,lmax);
   free_dvector(Smodes_A,0,lmax);
   free_dvector(sinMGtheta,0,ntheta);
   free_ivector(lambda_exp,0,ntheta);
}

/* sht_grid_synthesis_vector_1
 * *** TRANSFORMS FROM VECTOR SPHERICAL A_LM TO FUNC(THETA,PHI) ON EQUIRECTANGULAR GRID ***
 *
 * The E_lm, B_lm must be in the SPHERE_MODES->coefs convention
 * The grid output is:
 *
 * {v1,v2}map[x][y] += {v1,v2}(phi = x*delta, theta = y*delta)
 *
 * where delta = 2pi/nspace
 * and the output range of func_map is [xmin..xmax][ymin..ymax].
 * (Note: we use this range but if func_map is defined outside of this,
 * this will not cause a problem.)
 *
 * nspace MUST be a multiple of 4.  It is best for it to be a power of two, or a
 * power of two times a small odd number (this is for the FFT).  If nspace is 
 * a multiple of 4 with lots of odd factors, this routine will still work but will
 * be slow.
 * If convolve_flag!=0, ALM_V[L][M] is multiplied by convolve_kernel_V[L] prior
 * to the transform, and similarly for ALM_A and convolve_kernel_A.
 */

void sht_grid_synthesis_vector_1(double **V_LM, double **A_LM, double **v1map, double **v2map, long lmax, long nspace,
   long xmin, long xmax, long ymin, long ymax, double *convolve_kernel_V, double *convolve_kernel_A,
   unsigned short int convolve_flag) {

   int vsign;
   long nspace_coarse, nspace_max, quarter_nspace_max, two_nspace_max;
   double dtheta;
   double **FuncThetaPhi, **func_map;
   double **AMTheta_1, **AMTheta_2, **AMTheta;
   double **AMFine_1, **AMFine_2, **AMFine;
   double *AMFinePtrM, *AMFinePtrP;
   double *AMThetaPtrM, *AMThetaPtrP;
   double *FourierStackFine, *FourierStackCoarse;
   long two_nspace, half_nspace, quarter_nspace;
   long ntheta_m, two_ntheta_m, four_ntheta_m, eight_ntheta_m;
   long M,i,x,y,j;
   long xoffset, yoffset;
   int k;

   /* Check that we were given a multiple of 4 */
   if (nspace % 4) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld should be a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Compute useful quantities related to nspace */
   two_nspace = nspace << 1;
   half_nspace = nspace >> 1;
   quarter_nspace = nspace >> 2;

   /* Determine how finely we need to sample theta.  Here dtheta is the spacing of our points in theta and
    * ntheta is the total number spaced around the entire great circle meridian (2pi).
    * We also determine nspace_max and quarter_nspace_max; nspace_max will be the maximum
    * number of points we need to consider on the entire meridian, quarter_nspace_max will
    * be only for the quarter-meridian from 0 to pi/2.
    */
   nspace_max = 0;
   nspace_coarse = optimum_length_1( 2*lmax+1, BETA_FFT_SHT);
      /* The "+1" in the optimum_length_1 argument is there so that we do not have
       * any power at the Nyquist frequency itself (this would result in aliasing
       * and consequent loss of information).
       */
   ntheta_m = nspace_coarse >> 2;
   two_ntheta_m = ntheta_m << 1;
   four_ntheta_m = ntheta_m << 2;
   eight_ntheta_m = ntheta_m << 3;
   dtheta = TwoPi/nspace_coarse;
   nspace_max = nspace_coarse;
   quarter_nspace_max = nspace_max >> 2;
   two_nspace_max = nspace_max << 1;

   /* Check that we are over-sampling */
   if (nspace_coarse > nspace) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld, lmax=%ld, this is not oversampled.\n",nspace,lmax);
      exit(1);
   }

   /* Allocate and compute the a_M(theta)'s */
   AMTheta_1 = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   AMTheta_2 = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   AMFine_1 = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   AMFine_2 = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   FourierStackFine = dvector(0,two_nspace-1);
   FourierStackCoarse = dvector(0,two_nspace_max-1);
   latitude_syn_vector_1(V_LM,A_LM,AMTheta_1,AMTheta_2,dtheta,ntheta_m,lmax,convolve_kernel_V,
      convolve_kernel_A,convolve_flag);

   /* Loop over v1 and v2 for trig interpolation stage */
   for(k=0;k<2;k++) {
      AMTheta = (k==0)? AMTheta_1:AMTheta_2;
      AMFine = (k==0)? AMFine_1:AMFine_2;

      /* For each M, we do an exact trigonometric interpolation to yield
       * the values of am(theta) at a grid of points spaced in theta by
       * TwoPi/nspace.  The method is to consider a complete meridian
       * starting at the S pole, going up to the N pole, and then back
       * down at the opposite (+180deg) longitude.
       */
      for(M=0;M<=lmax;M++) {

         /* Set up the arrays for the latitude FFT */
         AMThetaPtrP = AMTheta[ M] - ntheta_m;
         AMThetaPtrM = AMTheta[-M] - ntheta_m;
         for(i=0;i<two_nspace_max;i++) FourierStackCoarse[i] = 0;
         FourierStackCoarse[0] = AMThetaPtrP[0] / four_ntheta_m;
         FourierStackCoarse[1] = AMThetaPtrM[0] / four_ntheta_m;

         /* the symmetry across theta => -theta is different for odd and even M.  The division by
          * four_ntheta_m is designed so that the forward and backward Fourier transforms will be
          * inverses of each other, without any multiplying factors.
          *
          * Note that the parity of the vector is opposite that of scalar or tensor (it has odd
          * spin), so the symmetry under theta => -theta has an extra negative sign, compare
          * to sht_grid_synthesis_1 and sht_grid_synthesis_polar_1.
          */
         if (M%2==1) { /* M odd */
            for(i=1;i<=two_ntheta_m;i++) {
               FourierStackCoarse[eight_ntheta_m - 2*i   ] =  (FourierStackCoarse[2*i  ] = AMThetaPtrP[i]/four_ntheta_m);
               FourierStackCoarse[eight_ntheta_m - 2*i +1] =  (FourierStackCoarse[2*i+1] = AMThetaPtrM[i]/four_ntheta_m);
            }
         } else { /* M even */
            for(i=1;i<=two_ntheta_m;i++) {
               FourierStackCoarse[eight_ntheta_m - 2*i   ] = -(FourierStackCoarse[2*i  ] = AMThetaPtrP[i]/four_ntheta_m);
               FourierStackCoarse[eight_ntheta_m - 2*i +1] = -(FourierStackCoarse[2*i+1] = AMThetaPtrM[i]/four_ntheta_m);
               /* We have set FourierStack[two_ntheta_m], FourierStack[two_ntheta_m+1] equal to
                * negative of their actual values.  This doesn't matter since they are zero anyway.
                */
            }
         }

         /* Now do the FFT to get the Fourier transform of a_M(theta) */
         fourier_trans_2(FourierStackCoarse-1,four_ntheta_m,-1);

         /* We will use FourierStackFine to do the forward FFT to get a_M(theta) at the new
          * points.  Begin by clearing it, then copying over positive, then negative frequency
          * modes, finally do the FFT itself.
          */
         for(i=0;i<two_nspace;i++) FourierStackFine[i] = 0;
         for(i=0;i<four_ntheta_m;i++) FourierStackFine[i] = FourierStackCoarse[i];
         for(i=1;i<=four_ntheta_m;i++) FourierStackFine[two_nspace - i] = FourierStackCoarse[eight_ntheta_m - i];
         fourier_trans_2(FourierStackFine-1,nspace,1);

         /* Put the interpolated values into AMFine */
         AMFinePtrP = AMFine[ M] - quarter_nspace;
         AMFinePtrM = AMFine[-M] - quarter_nspace;
         for(i=0;i<=half_nspace;i++) {
            AMFinePtrP[i] = FourierStackFine[2*i  ];
            AMFinePtrM[i] = FourierStackFine[2*i+1];
         }

      } /* end M loop */
   } /* end k={v1,v2} loop */

   /* Reallocate some memory */
   free_dmatrix(AMTheta_1,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   free_dmatrix(AMTheta_2,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   FuncThetaPhi = dmatrix(0,nspace-1,-quarter_nspace,quarter_nspace);

   /* Loop over {v1,v2} once again */
   for(k=0;k<2;k++) {
      AMFine = (k==0)? AMFine_1:AMFine_2;
      func_map = (k==0)? v1map:v2map;

      /* For each j, we do the longitude FFT */
      for(j=-quarter_nspace;j<=quarter_nspace;j++) {

         /* load elements of AMFine into FourierStack */
         for(i=0;i<two_nspace;i++) FourierStackFine[i]=0;
         FourierStackFine[0] = AMFine[0][j];
         for(M=1;M<=lmax;M++) {
            FourierStackFine[2*M] = AMFine[M][j];
            FourierStackFine[2*M+1]= AMFine[-M][j];
         }

         /* FFT from AM(theta) to A(phi)(theta) */
         fourier_trans_2(FourierStackFine-1,nspace,1);
         for(i=0;i<nspace;i++) FuncThetaPhi[i][j] = FourierStackFine[2*i];

      } /* end j loop */

      /* Now build the output map */
      xoffset = (xmin>=0)? 0 : nspace * ((-xmin)/nspace + 1);
      yoffset = (ymin>=0)? 0 : nspace * ((-ymin)/nspace + 1);
      for(x=xmin;x<=xmax;x++) {
         for(y=ymin;y<=ymax;y++) {
            vsign = 1;
            i = (x + xoffset) % nspace;
            j = (y + yoffset) % nspace;
            if (j>=3*quarter_nspace) j -= nspace;
            if (j>quarter_nspace) {
               i = (i + half_nspace) % nspace;
               j = half_nspace - j;
               vsign = -1;            /* Flips sign because vector components are odd under
                                       * theta --> -theta and phi --> phi+pi
                                       */
            }

            func_map[x][y] += FuncThetaPhi[i][j] * vsign;
         }
      }

   } /* end k={v1,v2} loop */

   /* Clean up memory */
   free_dmatrix(AMFine_1,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dmatrix(AMFine_2,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dvector(FourierStackFine,0,two_nspace-1);
   free_dvector(FourierStackCoarse,0,two_nspace_max-1);
   free_dmatrix(FuncThetaPhi,0,nspace-1,-quarter_nspace,quarter_nspace);

}

/* latitude_syn_polar_1
 * *** TRANSFORMS FROM POLARIZATION SPHERICAL HARMONIC A_LM TO A_M(THETA) ***
 *
 * This function performs the latitude part of a spherical harmonic transform.
 * The function is also capable of doing a convolution on the a_lm's before
 * doing the spherical harmonic transform.
 *
 * The storage conventions are:
 *
 * ALM_E, ALM_B = coefficients according to SPHERE_MODES->coefs
 * AMTheta[M][y] = Re a_M(pi/2 - y*dtheta) [M>=0]
 *                   Im a_|M|(pi/2 - y*dtheta) [M<0]
 *
 * func(theta,phi) = Re sum_{M>=0} a_M(theta) e^(iM phi)
 *
 * Dimensions: AMTheta_{Q,U}[-lmax..lmax][-ntheta..ntheta]
 * (Note: we use this range but if ALM is defined outside of this,
 * this will not cause a problem.)
 *
 * If convolve_flag!=0, ALM_E[L][M] is multiplied by convolve_kernel_E[L] prior
 * to the transform, and similarly for ALM_B and convolve_kernel_B.
 */

void latitude_syn_polar_1(double **ALM_E, double **ALM_B, double **AMTheta_Q,
   double **AMTheta_U, double dtheta, long ntheta, long lmax,
   double *convolve_kernel_E, double *convolve_kernel_B,
   unsigned short int convolve_flag) {

   long L,M,y,j;
   double *RecA, *RecB; /* recursion coefficients for generating Y_LM */
   double *RecAp, *RecBp;
   double *RecC, *RecD, *RecE;
   double *RecCp, *RecDp, *RecEp;
   double *Cmodes_E, *Smodes_E, *Cmodes_B, *Smodes_B;
   double *Cmp_E, *Smp_E, *Cmp_B, *Smp_B;
   double costheta, sintheta;
   double AbsWLM, AbsXLM, sin2theta, multAbsYLm1M;
   double *sinMGtheta;
   int *lambda_exp;
   int lambda_exp_current;
   double InitializerCoef;
   double normalization;
   double AbsYLp1M,AbsYLM,AbsYLm1M;
   double AMCEven_Q, AMCOdd_Q, AMSEven_Q, AMSOdd_Q;
   double AMCEven_U, AMCOdd_U, AMSEven_U, AMSOdd_U;
   unsigned short int latitude_parity;

   /* Allocate auxiliary arrays */
   RecA = dvector(0,lmax);
   RecB = dvector(0,lmax);
   RecC = dvector(0,lmax);
   RecD = dvector(0,lmax);
   RecE = dvector(0,lmax);
   Cmodes_E = dvector(0,lmax);
   Smodes_E = dvector(0,lmax);
   Cmodes_B = dvector(0,lmax);
   Smodes_B = dvector(0,lmax);
   sinMGtheta = dvector(0,ntheta);
   lambda_exp = ivector(0,ntheta);

   /* Initialize sin^(M-G)(theta) array and exponent overflow array */
   for(y=0;y<=ntheta;y++) {
      sinMGtheta[y] = 1.;
      lambda_exp[y] = 0;
   }

   /* Outer loop over the M-values */
   for(M=0;M<=lmax;M++) {

      /* Initialize recurrance relation coefficients */
      for(L=M;L<=lmax;L++) {

         /* Y recursions */
         RecA[L] = sqrt((2*L+3.0)/((L+1)*(L+1)-M*M));
         RecB[L] = sqrt((L*L-M*M)/(2*L-1.0))*RecA[L];
         RecA[L] *= sqrt(2*L+1.0);

         /* W and X recursions -- these depend on Y. */
         RecC[L] = (double)(L-M*M);
         RecD[L] = 0.5*(L*(L-1));
         RecE[L] = (L>0)? sqrt( (L*L-M*M) * ((double)(2*L+1)/(2*L-1)) ) : 0;
      }
      InitializerCoef=OneOverTwoPi;
      for(j=1;j<=M;j++) InitializerCoef *= 1.0 + 0.5/j;
      InitializerCoef = sqrt(InitializerCoef);
      if (M%2==1) InitializerCoef *= -1;

      /* and construct the vectors of cosinelike and sinelike coefficients */
      for(L=M;L<=lmax;L++) {
         Cmodes_E[L] = ALM_E[L][M];
         Smodes_E[L] = ALM_E[L][-M];
         Cmodes_B[L] = ALM_B[L][M];
         Smodes_B[L] = ALM_B[L][-M];
      }
      if (M==0) for(L=0;L<=lmax;L++) {
         Smodes_E[L] = Smodes_B[L] = 0; /* no sin(0phi) mode */
         Cmodes_E[L] /= SqrtTwo;
         Cmodes_B[L] /= SqrtTwo;
      }

      /* do the pre-transform convolution, if necessary */
      if (convolve_flag) {
         for(L=M;L<=lmax;L++) {
            Cmodes_E[L] *= convolve_kernel_E[L];
            Smodes_E[L] *= convolve_kernel_E[L];
            Cmodes_B[L] *= convolve_kernel_B[L];
            Smodes_B[L] *= convolve_kernel_B[L];
         }
      }

      /* and now apply the normalization factor, i.e. the conversion
       * between (W+iX)_lm and (Q+iU).  Note that we are using the EB
       * convention, as distinct from the GC convention, hence our
       * normalization factor is different from Kamionkowski et al
       * N_l (see astro-ph/9611125v1 eq 2.16) by a factor of sqrt2.
       * Notice that we wipe out modes with L<2.
       */
      for(L=M;L<=lmax;L++) {
         normalization = (L>=2)? 2./sqrt(((double)(L-1)*L)*(L+1)*(L+2)) :0 ;
         Cmodes_E[L] *= normalization;
         Smodes_E[L] *= normalization;
         Cmodes_B[L] *= normalization;
         Smodes_B[L] *= normalization;
      }

      /* Now run through all the points */
      for(y=0;y<=ntheta;y++) {
         /* Note that y*dtheta is the latitude but theta is the colatitude;
          * this is why sin and cos look backward but yes, this is correct.
          */
         costheta = sin(y*dtheta);
         sintheta = cos(y*dtheta);
         sin2theta = sintheta * sintheta;

         /* Initialize the spherical harmonics at L=|M|; we will   
          * use the recurrance relation, based on NR 6.8.2,7 to    
          * find the normalized latitude part of the spherical     
          * harmonic.  The recurrance is:                          
          * sqrt((L^2-M^2)/(2L+1) * |Y_L_M| =
          *                        sqrt(2L-1) * cos(theta)|Y_L-1_M|
          *     - sqrt(((L-1)^2-M^2)/(2L-3)) * |Y_L-2_M|           
          * AbsYLMs are normalized to Int(AbsYLM sin^G theta)^2 dcostheta = 1/Pi
          */
         if (sinMGtheta[y] < LAMBDA_MIN) {
            sinMGtheta[y] *= INV_LAMBDA_MIN;
            lambda_exp[y]++;
         }
         lambda_exp_current = lambda_exp[y];
         AbsYLm1M = 0;
         AbsYLM = InitializerCoef * sinMGtheta[y];

         /* Increment sinMGtheta for next use */
         if (M>=2) sinMGtheta[y] *= sintheta;

         /* Initialize pointers */
         RecAp = RecA + M;
         RecBp = RecB + M;
         RecCp = RecC + M;
         RecDp = RecD + M;
         RecEp = RecE + M;
         Cmp_E = Cmodes_E + M;
         Smp_E = Smodes_E + M;
         Cmp_B = Cmodes_B + M;
         Smp_B = Smodes_B + M;

         /* Initialize the opposite-parity arrays and parity counter */
         AMCEven_Q = AMCOdd_Q = AMSEven_Q = AMSOdd_Q = 0.;
         AMCEven_U = AMCOdd_U = AMSEven_U = AMSOdd_U = 0.;
         latitude_parity = 0;

         /* Now go through the values of L */
         for(L=M;L<=lmax;L++) {

            if (lambda_exp_current>0) {
               /* If lambda_exp_current>0 we don't need to do the
                * multiplication/addition of the output, but we must
                * see whether Ylm is large enough that we can reduce
                * the number of orders of underflow, lambda_exp_current.
                */
               if (fabs(AbsYLM) > 1.0) {
                  lambda_exp_current--;
                  AbsYLM *= LAMBDA_MIN;
                  AbsYLm1M *= LAMBDA_MIN;
               }

            }

            if (lambda_exp_current==0) {

               /* First we need to compute W and X.  Note that AbsWLM and
                * AbsXLM are actually (sin^2theta)/2 times W and X.  The factor
                * of 2 is accounted for in the definitions of YG and YC, whereas
                * the factor of sin^(2-G)(theta) is inserted after the sum.
                */
               multAbsYLm1M = (*RecEp) * AbsYLm1M;
               AbsWLM = (*RecCp + (*RecDp) * sin2theta) * AbsYLM - costheta * multAbsYLm1M;
               AbsXLM = M*( (L-1) * costheta * AbsYLM - multAbsYLm1M );

               /* Do the multiplication for each latitude ring
                * Note that we only increment the even or odd parity
                * a_m(theta)'s depending on the parity of the
                * particular Ylm.
                */
               if (latitude_parity) {
                  AMCOdd_Q   += AbsWLM * (*Cmp_E);
                  AMSOdd_Q   += AbsWLM * (*Smp_E);
                  AMSEven_Q  += AbsXLM * (*Cmp_B);
                  AMCEven_Q  -= AbsXLM * (*Smp_B);
                  AMCOdd_U   += AbsWLM * (*Cmp_B);
                  AMSOdd_U   += AbsWLM * (*Smp_B);
                  AMSEven_U  -= AbsXLM * (*Cmp_E);
                  AMCEven_U  += AbsXLM * (*Smp_E);
               } else {
                  AMCEven_Q  += AbsWLM * (*Cmp_E);
                  AMSEven_Q  += AbsWLM * (*Smp_E);
                  AMSOdd_Q   += AbsXLM * (*Cmp_B);
                  AMCOdd_Q   -= AbsXLM * (*Smp_B);
                  AMCEven_U  += AbsWLM * (*Cmp_B);
                  AMSEven_U  += AbsWLM * (*Smp_B);
                  AMSOdd_U   -= AbsXLM * (*Cmp_E);
                  AMCOdd_U   += AbsXLM * (*Smp_E);
               }
            }

            /* and then increment the Ylm and pointers */
            AbsYLp1M = (*RecAp) * costheta * AbsYLM - (*RecBp) * AbsYLm1M;
            RecAp++; RecBp++; RecCp++; RecDp++; RecEp++;
            Cmp_E++; Smp_E++; Cmp_B++; Smp_B++;
            AbsYLm1M = AbsYLM;
            AbsYLM = AbsYLp1M;

            latitude_parity ^= 1; /* switches between 0 and 1. */

         } /* end L loop */

         /* This block of code will add our results to the
          * output arrays.  This step is unnecessary (in fact, the code
          * will crash!) if M<2 and sintheta==0, hence we block that case.
          */
         if (M==2 || fabs(sintheta)>MIN_SIN_THETA) {

            /* Divide the resulting Q and U by sin^(2-G)(theta) if necessary */
            if (M==0) {
               AMCEven_Q /= sin2theta;
               AMCEven_U /= sin2theta;
               AMCOdd_Q  /= sin2theta;
               AMCOdd_U  /= sin2theta;
            }
            if (M==1) {
               AMCEven_Q /= sintheta;
               AMCEven_U /= sintheta;
               AMCOdd_Q  /= sintheta;
               AMCOdd_U  /= sintheta;
               AMSEven_Q /= sintheta;
               AMSEven_U /= sintheta;
               AMSOdd_Q  /= sintheta;
               AMSOdd_U  /= sintheta;
            }

            /* Now put these results into the output arrays */
            AMTheta_Q[ M][ y] = AMCEven_Q + AMCOdd_Q;
            AMTheta_Q[ M][-y] = AMCEven_Q - AMCOdd_Q;
            AMTheta_U[ M][ y] = AMCEven_U + AMCOdd_U;
            AMTheta_U[ M][-y] = AMCEven_U - AMCOdd_U;
            if (M!=0) {
               AMTheta_Q[-M][ y] = AMSEven_Q + AMSOdd_Q;
               AMTheta_Q[-M][-y] = AMSEven_Q - AMSOdd_Q;
               AMTheta_U[-M][ y] = AMSEven_U + AMSOdd_U;
               AMTheta_U[-M][-y] = AMSEven_U - AMSOdd_U;
            }
         }

      } /* end y loop */

   } /* end M loop */

   /* De-allocate auxiliary arrays */
   free_dvector(RecA,0,lmax);
   free_dvector(RecB,0,lmax);
   free_dvector(RecC,0,lmax);
   free_dvector(RecD,0,lmax);
   free_dvector(RecE,0,lmax);
   free_dvector(Cmodes_E,0,lmax);
   free_dvector(Smodes_E,0,lmax);
   free_dvector(Cmodes_B,0,lmax);
   free_dvector(Smodes_B,0,lmax);
   free_dvector(sinMGtheta,0,ntheta);
   free_ivector(lambda_exp,0,ntheta);
}

/* sht_grid_synthesis_polar_1
 * *** TRANSFORMS FROM POLARIZATION SPHERICAL A_LM TO FUNC(THETA,PHI) ON EQUIRECTANGULAR GRID ***
 *
 * The E_lm, B_lm must be in the SPHERE_MODES->coefs convention
 * The grid output is:
 *
 * {Q,U}map[x][y] += {Q,U}(phi = x*delta, theta = y*delta)
 *
 * where delta = 2pi/nspace
 * and the output range of func_map is [xmin..xmax][ymin..ymax].
 * (Note: we use this range but if func_map is defined outside of this,
 * this will not cause a problem.)
 *
 * nspace MUST be a multiple of 4.  It is best for it to be a power of two, or a
 * power of two times a small odd number (this is for the FFT).  If nspace is 
 * a multiple of 4 with lots of odd factors, this routine will still work but will
 * be slow.
 * If convolve_flag!=0, ALM_E[L][M] is multiplied by convolve_kernel_E[L] prior
 * to the transform, and similarly for ALM_B and convolve_kernel_B.
 */

void sht_grid_synthesis_polar_1(double **E_LM, double **B_LM, double **Qmap, double **Umap, long lmax, long nspace,
   long xmin, long xmax, long ymin, long ymax, double *convolve_kernel_E, double *convolve_kernel_B,
   unsigned short int convolve_flag) {

   long nspace_coarse, nspace_max, quarter_nspace_max, two_nspace_max;
   double dtheta;
   double **FuncThetaPhi, **func_map;
   double **AMTheta_Q, **AMTheta_U, **AMTheta;
   double **AMFine_Q, **AMFine_U, **AMFine;
   double *AMFinePtrM, *AMFinePtrP;
   double *AMThetaPtrM, *AMThetaPtrP;
   double *FourierStackFine, *FourierStackCoarse;
   long two_nspace, half_nspace, quarter_nspace;
   long ntheta_m, two_ntheta_m, four_ntheta_m, eight_ntheta_m;
   long M,i,x,y,j;
   long xoffset, yoffset;
   int k;

   /* Check that we were given a multiple of 4 */
   if (nspace % 4) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld should be a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Compute useful quantities related to nspace */
   two_nspace = nspace << 1;
   half_nspace = nspace >> 1;
   quarter_nspace = nspace >> 2;

   /* Determine how finely we need to sample theta.  Here dtheta is the spacing of our points in theta and
    * ntheta is the total number spaced around the entire great circle meridian (2pi).
    * We also determine nspace_max and quarter_nspace_max; nspace_max will be the maximum
    * number of points we need to consider on the entire meridian, quarter_nspace_max will
    * be only for the quarter-meridian from 0 to pi/2.
    */
   nspace_max = 0;
   nspace_coarse = optimum_length_1( 2*lmax+1, BETA_FFT_SHT);
      /* The "+1" in the optimum_length_1 argument is there so that we do not have
       * any power at the Nyquist frequency itself (this would result in aliasing
       * and consequent loss of information).
       */
   ntheta_m = nspace_coarse >> 2;
   two_ntheta_m = ntheta_m << 1;
   four_ntheta_m = ntheta_m << 2;
   eight_ntheta_m = ntheta_m << 3;
   dtheta = TwoPi/nspace_coarse;
   nspace_max = nspace_coarse;
   quarter_nspace_max = nspace_max >> 2;
   two_nspace_max = nspace_max << 1;

   /* Check that we are over-sampling */
   if (nspace_coarse > nspace) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld, lmax=%ld, this is not oversampled.\n",nspace,lmax);
      exit(1);
   }

   /* Allocate and compute the a_M(theta)'s */
   AMTheta_Q = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   AMTheta_U = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   AMFine_Q = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   AMFine_U = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   FourierStackFine = dvector(0,two_nspace-1);
   FourierStackCoarse = dvector(0,two_nspace_max-1);
   latitude_syn_polar_1(E_LM,B_LM,AMTheta_Q,AMTheta_U,dtheta,ntheta_m,lmax,convolve_kernel_E,
      convolve_kernel_B,convolve_flag);

   /* Loop over Q and U for trig interpolation stage */
   for(k=0;k<2;k++) {
      AMTheta = (k==0)? AMTheta_Q:AMTheta_U;
      AMFine = (k==0)? AMFine_Q:AMFine_U;

      /* For each M, we do an exact trigonometric interpolation to yield
       * the values of am(theta) at a grid of points spaced in theta by
       * TwoPi/nspace.  The method is to consider a complete meridian
       * starting at the S pole, going up to the N pole, and then back
       * down at the opposite (+180deg) longitude.
       */
      for(M=0;M<=lmax;M++) {

         /* Set up the arrays for the latitude FFT */
         AMThetaPtrP = AMTheta[ M] - ntheta_m;
         AMThetaPtrM = AMTheta[-M] - ntheta_m;
         for(i=0;i<two_nspace_max;i++) FourierStackCoarse[i] = 0;
         FourierStackCoarse[0] = AMThetaPtrP[0] / four_ntheta_m;
         FourierStackCoarse[1] = AMThetaPtrM[0] / four_ntheta_m;

         /* the symmetry across theta => -theta is different for odd and even M.  The division by
          * four_ntheta_m is designed so that the forward and backward Fourier transforms will be
          * inverses of each other, without any multiplying factors.
          */
         if (M%2==0) { /* M even */
            for(i=1;i<=two_ntheta_m;i++) {
               FourierStackCoarse[eight_ntheta_m - 2*i   ] =  (FourierStackCoarse[2*i  ] = AMThetaPtrP[i]/four_ntheta_m);
               FourierStackCoarse[eight_ntheta_m - 2*i +1] =  (FourierStackCoarse[2*i+1] = AMThetaPtrM[i]/four_ntheta_m);
            }
         } else { /* M odd */
            for(i=1;i<=two_ntheta_m;i++) {
               FourierStackCoarse[eight_ntheta_m - 2*i   ] = -(FourierStackCoarse[2*i  ] = AMThetaPtrP[i]/four_ntheta_m);
               FourierStackCoarse[eight_ntheta_m - 2*i +1] = -(FourierStackCoarse[2*i+1] = AMThetaPtrM[i]/four_ntheta_m);
               /* We have set FourierStack[two_ntheta_m], FourierStack[two_ntheta_m+1] equal to
                * negative of their actual values.  This doesn't matter since they are zero anyway.
                */
            }
         }

         /* Now do the FFT to get the Fourier transform of a_M(theta) */
         fourier_trans_2(FourierStackCoarse-1,four_ntheta_m,-1);

         /* We will use FourierStackFine to do the forward FFT to get a_M(theta) at the new
          * points.  Begin by clearing it, then copying over positive, then negative frequency
          * modes, finally do the FFT itself.
          */
         for(i=0;i<two_nspace;i++) FourierStackFine[i] = 0;
         for(i=0;i<four_ntheta_m;i++) FourierStackFine[i] = FourierStackCoarse[i];
         for(i=1;i<=four_ntheta_m;i++) FourierStackFine[two_nspace - i] = FourierStackCoarse[eight_ntheta_m - i];
         fourier_trans_2(FourierStackFine-1,nspace,1);

         /* Put the interpolated values into AMFine */
         AMFinePtrP = AMFine[ M] - quarter_nspace;
         AMFinePtrM = AMFine[-M] - quarter_nspace;
         for(i=0;i<=half_nspace;i++) {
            AMFinePtrP[i] = FourierStackFine[2*i  ];
            AMFinePtrM[i] = FourierStackFine[2*i+1];
         }

      } /* end M loop */
   } /* end k={Q,U} loop */

   /* Reallocate some memory */
   free_dmatrix(AMTheta_Q,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   free_dmatrix(AMTheta_U,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   FuncThetaPhi = dmatrix(0,nspace-1,-quarter_nspace,quarter_nspace);

   /* Loop over {Q,U} once again */
   for(k=0;k<2;k++) {
      AMFine = (k==0)? AMFine_Q:AMFine_U;
      func_map = (k==0)? Qmap:Umap;

      /* For each j, we do the longitude FFT */
      for(j=-quarter_nspace;j<=quarter_nspace;j++) {

         /* load elements of AMFine into FourierStack */
         for(i=0;i<two_nspace;i++) FourierStackFine[i]=0;
         FourierStackFine[0] = AMFine[0][j];
         for(M=1;M<=lmax;M++) {
            FourierStackFine[2*M] = AMFine[M][j];
            FourierStackFine[2*M+1]= AMFine[-M][j];
         }

         /* FFT from AM(theta) to A(phi)(theta) */
         fourier_trans_2(FourierStackFine-1,nspace,1);
         for(i=0;i<nspace;i++) FuncThetaPhi[i][j] = FourierStackFine[2*i];

      } /* end j loop */

      /* Now build the output map */
      xoffset = (xmin>=0)? 0 : nspace * ((-xmin)/nspace + 1);
      yoffset = (ymin>=0)? 0 : nspace * ((-ymin)/nspace + 1);
      for(x=xmin;x<=xmax;x++) {
         for(y=ymin;y<=ymax;y++) {
            i = (x + xoffset) % nspace;
            j = (y + yoffset) % nspace;
            if (j>=3*quarter_nspace) j -= nspace;
            if (j>quarter_nspace) {
               i = (i + half_nspace) % nspace;
               j = half_nspace - j;
            }

            func_map[x][y] += FuncThetaPhi[i][j];
         }
      }

   } /* end k={Q,U} loop */

   /* Clean up memory */
   free_dmatrix(AMFine_Q,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dmatrix(AMFine_U,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dvector(FourierStackFine,0,two_nspace-1);
   free_dvector(FourierStackCoarse,0,two_nspace_max-1);
   free_dmatrix(FuncThetaPhi,0,nspace-1,-quarter_nspace,quarter_nspace);

}

/* latitude_ana_1
 * *** TRANSFORMS FROM A_M(THETA) TO SPHERICAL HARMONIC A_LM ***
 *
 * This function performs the latitude part of a spherical harmonic transform; the function
 * is also capable of doing a convolution on the a_lm's after doing the spherical
 * harmonic transform.  It is the transpose (NOT the inverse!!!) of latitude_syn_1.
 * It works by doing the transpose of all of the linear operations of latitude_syn_1 in
 * reverse order.
 *
 * The storage conventions are:
 *
 * ALM = coefficients according to SPHERE_MODES->coefs
 * AMTheta[M][y] = Re a_M(pi/2 - y*dtheta) [M>=0]
 *                 Im a_|M|(pi/2 - y*dtheta) [M<0]
 *
 * func(theta,phi) = Re sum_{M>=0} a_M(theta) e^(iM phi) sin^G(theta)
 *
 * Dimensions: AMTheta[-lmax..lmax][-ntheta..ntheta]
 * (Note: we use this range but if ALM is defined outside of this,
 * this will not cause a problem.)
 *
 * If convolve_flag!=0, ALM[L][M] is multiplied by convolve_kernel[L] after
 * the transform.
 */

void latitude_ana_1(double **ALM, double **AMTheta, double dtheta, long ntheta, long lmax,
   double *convolve_kernel, unsigned short int convolve_flag) {

   long L,M,y,j;
   double *RecA, *RecB; /* recursion coefficients for generating Y_LM */
   double *RecAp, *RecBp;
   double *Cmodes, *Smodes;
   double *Cmp, *Smp;
   double costheta, sintheta;
   double *sinMGtheta;
   int *lambda_exp;
   int lambda_exp_current;
   double InitializerCoef;
   double AbsYLp1M,AbsYLM,AbsYLm1M;
   double AMCEven, AMCOdd, AMSEven, AMSOdd;
   unsigned short int latitude_parity;

   /* Allocate auxiliary arrays */
   RecA = dvector(0,lmax);
   RecB = dvector(0,lmax);
   Cmodes = dvector(0,lmax);
   Smodes = dvector(0,lmax);
   sinMGtheta = dvector(0,ntheta);
   lambda_exp = ivector(0,ntheta);

   /* Initialize sin^(M-G)(theta) array and exponent overflow array */
   for(y=0;y<=ntheta;y++) {
      sinMGtheta[y] = 1.;
      lambda_exp[y] = 0;
   }

   /* Outer loop over the M-values */
   for(M=0;M<=lmax;M++) {

      /* Initialize recurrance relation coefficients */
      for(L=M;L<=lmax;L++) {
         RecA[L] = sqrt((2*L+3.0)/((L+1)*(L+1)-M*M));
         RecB[L] = sqrt((L*L-M*M)/(2*L-1.0))*RecA[L];
         RecA[L] *= sqrt(2*L+1.0);
      }
      InitializerCoef=OneOverTwoPi;
      for(j=1;j<=M;j++) InitializerCoef *= 1.0 + 0.5/j;
      InitializerCoef = sqrt(InitializerCoef);
      if (M%2==1) InitializerCoef *= -1;

      /* Initialize the Cmp and Smp arrays, which are going to be the outputs,
       * i.e. Cmodes[L] will get stored to ALM[L][M] and Smodes[L] will get
       * stored to ALM[L][-M].
       */
      for(L=M;L<=lmax;L++) Cmodes[L] = Smodes[L] = 0.;

      /* Now run through all the points */
      for(y=0;y<=ntheta;y++) {

         /* Get the even and odd transform values */
         if (y>0) {
            AMCEven = AMTheta[ M][ y] + AMTheta[ M][-y];
            AMCOdd  = AMTheta[ M][ y] - AMTheta[ M][-y];
            if (M!=0) {
               AMSEven = AMTheta[-M][ y] + AMTheta[-M][-y];
               AMSOdd  = AMTheta[-M][ y] - AMTheta[-M][-y];
            } else {
               AMSEven = AMSOdd = 0.;
            }
         } else { /* y==0 */
            AMCEven = AMTheta[ M][0];
            if (M!=0) {
               AMSEven = AMTheta[-M][0];
            } else {
               AMSEven = 0.;
            }
            AMCOdd = AMSOdd = 0.;
         }

         /* Note that y*dtheta is the latitude but theta is the colatitude;
          * this is why sin and cos look backward but yes, this is correct.
          */
         costheta = sin(y*dtheta);
         sintheta = cos(y*dtheta);

         /* Initialize the spherical harmonics at L=|M|; we will   
          * use the recurrance relation, based on NR 6.8.2,7 to    
          * find the normalized latitude part of the spherical     
          * harmonic.  The recurrance is:                          
          * sqrt((L^2-M^2)/(2L+1) * |Y_L_M| =
          *                        sqrt(2L-1) * cos(theta)|Y_L-1_M|
          *     - sqrt(((L-1)^2-M^2)/(2L-3)) * |Y_L-2_M|           
          * AbsYLMs are normalized to Int(AbsYLM sin^G theta)^2 dcostheta = 1/Pi
          */
         if (sinMGtheta[y] < LAMBDA_MIN) {
            sinMGtheta[y] *= INV_LAMBDA_MIN;
            lambda_exp[y]++;
         }
         lambda_exp_current = lambda_exp[y];
         AbsYLm1M = 0;
         AbsYLM = InitializerCoef * sinMGtheta[y];

         /* Increment sinMGtheta for next use */
         sinMGtheta[y] *= sintheta;

         /* Initialize pointers */
         RecAp = RecA + M;
         RecBp = RecB + M;
         Cmp = Cmodes + M;
         Smp = Smodes + M;

         /* Initialize the parity counter */
         latitude_parity = 0;

         /* Now go through the values of L */
         for(L=M;L<=lmax;L++) {

            if (lambda_exp_current>0) {
               /* If lambda_exp_current>0 we don't need to do the
                * multiplication/addition of the output, but we must
                * see whether Ylm is large enough that we can reduce
                * the number of orders of underflow, lambda_exp_current.
                */
               if (fabs(AbsYLM) > 1.0) {
                  lambda_exp_current--;
                  AbsYLM *= LAMBDA_MIN;
                  AbsYLm1M *= LAMBDA_MIN;
               }

            }

            if (lambda_exp_current==0) {
               /* Do the multiplication for each latitude ring
                * Note that we read the even or odd parity
                * a_m(theta)'s depending on the parity of the
                * particular Ylm.
                */
               if (latitude_parity) {
                  *Cmp  += AbsYLM * AMCOdd;
                  *Smp  += AbsYLM * AMSOdd;
               } else {
                  *Cmp  += AbsYLM * AMCEven;
                  *Smp  += AbsYLM * AMSEven;
               }

            }

            /* and then increment the Ylm and pointers */
            AbsYLp1M = (*RecAp) * costheta * AbsYLM - (*RecBp) * AbsYLm1M;
            RecAp++; RecBp++;
            Cmp++; Smp++;
            AbsYLm1M = AbsYLM;
            AbsYLM = AbsYLp1M;

            latitude_parity ^= 1; /* switches between 0 and 1. */

         } /* end L loop */

      } /* end y loop */

      /* do the post-transform convolution, if necessary */
      if (convolve_flag) {
         for(L=M;L<=lmax;L++) {
            Cmodes[L] *= convolve_kernel[L];
            Smodes[L] *= convolve_kernel[L];
         }
      }

      /* and construct the vectors of cosinelike and sinelike coefficients */
      if (M==0) for(L=0;L<=lmax;L++) {
         Smodes[L] = 0; /* no sin(0phi) mode */
         Cmodes[L] /= SqrtTwo;
      }
      for(L=M;L<=lmax;L++) {
         ALM[L][-M] += Smodes[L];
         ALM[L][ M] += Cmodes[L];
      }

   } /* end M loop */

   /* De-allocate auxiliary arrays */
   free_dvector(RecA,0,lmax);
   free_dvector(RecB,0,lmax);
   free_dvector(Cmodes,0,lmax);
   free_dvector(Smodes,0,lmax);
   free_dvector(sinMGtheta,0,ntheta);
   free_ivector(lambda_exp,0,ntheta);
}

/* sht_grid_analysis_1
 * *** TRANSFORMS FROM FUNC(THETA,PHI) TO SPHERICAL A_LM ON EQUIRECTANGULAR GRID ***
 *
 * This function is the transpose (NOT the inverse!) of sht_grid_synthesis_1.  Unsurprisingly,
 * it works by doing all the same linear operations as sht_grid_synthesis_1, except that
 * they are transposed and are in reverse order.
 *
 * The a_lm must be in the SPHERE_MODES->coefs convention
 * The grid input is:
 *
 * func_map[x][y] = f(phi = x*delta, theta = y*delta)
 *
 * where delta = 2pi/nspace
 * and the input range of func_map is [xmin..xmax][ymin..ymax].
 * (Note: we use this range but if func_map is defined outside of this,
 * this will not cause a problem.)
 *
 * nspace MUST be a multiple of 4.  It is best for it to be a power of two, or a
 * power of two times a small odd number (this is for the FFT).  If nspace is 
 * a multiple of 4 with lots of odd factors, this routine will still work but will
 * be slow.
 *
 * Note that the output is ADDED TO A_LM, NOT over-written.  If you want the latter you must
 * initialize A_LM before calling this routine.
 *
 * If convolve_flag!=0, the function is multiplied by the convolve_kernel after
 * being transformed but before being added to A_LM.
 */

void sht_grid_analysis_1(double **ALM, double **func_map, long lmax, long nspace,
   long xmin, long xmax, long ymin, long ymax, double *convolve_kernel,
   unsigned short int convolve_flag) {

   long nspace_coarse, nspace_max, quarter_nspace_max, two_nspace_max;
   double dtheta;
   double **FuncThetaPhi;
   double **AMTheta;
   double **AMFine;
   double *AMFinePtrM, *AMFinePtrP;
   double *AMThetaPtrM, *AMThetaPtrP;
   double *FourierStackFine, *FourierStackCoarse;
   long two_nspace, half_nspace, quarter_nspace;
   long ntheta_m, two_ntheta_m, four_ntheta_m, eight_ntheta_m;
   long M,i,x,y,j;
   long xoffset, yoffset;

   /* First do some preliminary stuff, computing the dimensions for various arrays, etc. */

   /* Check that we were given a multiple of 4 */
   if (nspace % 4) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld should be a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Compute useful quantities related to nspace */
   two_nspace = nspace << 1;
   half_nspace = nspace >> 1;
   quarter_nspace = nspace >> 2;

   /* Determine how finely we need to sample theta.  Here dtheta is the spacing of our points in theta and
    * ntheta is the total number spaced around the entire great circle meridian (2pi).
    * We also determine nspace_max and quarter_nspace_max; nspace_max will be the maximum
    * number of points we need to consider on the entire meridian, quarter_nspace_max will
    * be only for the quarter-meridian from 0 to pi/2.
    */
   nspace_max = 0;
   nspace_coarse = optimum_length_1( 2*lmax+1, BETA_FFT_SHT);
      /* The "+1" in the optimum_length_1 argument is there so that we do not have
       * any power at the Nyquist frequency itself (this would result in aliasing
       * and consequent loss of information).
       */
   ntheta_m = nspace_coarse >> 2;
   two_ntheta_m = ntheta_m << 1;
   four_ntheta_m = ntheta_m << 2;
   eight_ntheta_m = ntheta_m << 3;
   dtheta = TwoPi/nspace_coarse;
   nspace_max = nspace_coarse;
   quarter_nspace_max = nspace_max >> 2;
   two_nspace_max = nspace_max << 1;

   /* Check that we are over-sampling */
   if (nspace_coarse > nspace) {
      fprintf(stderr,"Error in sht_grid_analysis_1: nspace=%ld, lmax=%ld, this is not oversampled.\n",nspace,lmax);
      exit(1);
   }

   /* This is where we begin doing the sht_grid_synthesis_1 operations in reverse order.
    * We begin by allocating some memory for the longitude transform.
    */
   FuncThetaPhi = dmatrix(0,nspace-1,-quarter_nspace,quarter_nspace);
   AMFine = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   FourierStackFine = dvector(0,two_nspace-1);
   FourierStackCoarse = dvector(0,two_nspace_max-1);

   /* Clear FuncThetaPhi and AMFine */
   for(i=0;i<nspace;i++) {
      for(j=-quarter_nspace;j<=quarter_nspace;j++) {
         FuncThetaPhi[i][j] = 0.;
   }}
   for(i= -lmax;i<=lmax;i++) {
      for(j=-quarter_nspace;j<=quarter_nspace;j++) {
         AMFine[i][j] = 0.;
   }}

   /* Read the output map func_map into FuncThetaPhi */
   xoffset = (xmin>=0)? 0 : nspace * ((-xmin)/nspace + 1);
   yoffset = (ymin>=0)? 0 : nspace * ((-ymin)/nspace + 1);
   for(x=xmin;x<=xmax;x++) {
      for(y=ymin;y<=ymax;y++) {
         i = (x + xoffset) % nspace;
         j = (y + yoffset) % nspace;
         if (j>=3*quarter_nspace) j -= nspace;
         if (j>quarter_nspace) {
            i = (i + half_nspace) % nspace;
            j = half_nspace - j;
         }

      FuncThetaPhi[i][j] += func_map[x][y];
      }
   }

   /* For each j, we do the longitude FFT */
   for(j=-quarter_nspace;j<=quarter_nspace;j++) {

      /* load elements of AMFine into FourierStack */
      for(i=0;i<two_nspace;i++) FourierStackFine[i]=0;
      for(i=0;i<nspace;i++) FourierStackFine[2*i] = FuncThetaPhi[i][j];

      /* FFT from A(phi)(theta) to AM(theta) -- notice that we use reverse FFT,
       * i.e. the isign=-1 option is used here.  (We want to use the Hermitian
       * conjugate of the FFT matrix, this is the reverse FFT.)
       */
      fourier_trans_2(FourierStackFine-1,nspace,-1);

      AMFine[0][j] = FourierStackFine[0];
      for(M=1;M<=lmax;M++) {
         AMFine[ M][j] = FourierStackFine[2*M];
         AMFine[-M][j] = FourierStackFine[2*M+1];
      }

   } /* end j loop */

   /* Reallocate some memory */
   AMTheta = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   free_dmatrix(FuncThetaPhi,0,nspace-1,-quarter_nspace,quarter_nspace);

   /* For each M, we do an exact trigonometric interpolation to yield
    * the values of am(theta) at a grid of points spaced in theta by
    * TwoPi/nspace_coarse.  The method is to consider a complete meridian
    * starting at the S pole, going up to the N pole, and then back
    * down at the opposite (+180deg) longitude.
    */
   for(M=0;M<=lmax;M++) {

      /* We will use FourierStackFine to do the reverse (transpose) FFT to take a_M(theta) at the
       * finely sampled points, and put this information on the coarse-grid.  Begin by clearing
       * it, then copying over positive, then negative frequency modes, finally do the FFT itself.
       */

      /* Put the AMFine values into the Fourier stack to be transformed */
      AMFinePtrP = AMFine[ M] - quarter_nspace;
      AMFinePtrM = AMFine[-M] - quarter_nspace;
      for(i=0;i<two_nspace;i++) FourierStackFine[i] = 0;
      for(i=0;i<=half_nspace;i++) {
         FourierStackFine[2*i  ] = AMFinePtrP[i];
         if (M!=0) FourierStackFine[2*i+1] = AMFinePtrM[i];
      }

      /* Transpose FFT, then copy the relevant Fourier modes over to FourierStackCoarse */
      fourier_trans_2(FourierStackFine-1,nspace,-1);
      for(i=0;i<two_nspace_max;i++) FourierStackCoarse[i] = 0;
      for(i=0;i<four_ntheta_m;i++) FourierStackCoarse[i] = FourierStackFine[i];
      for(i=1;i<=four_ntheta_m;i++) FourierStackCoarse[eight_ntheta_m - i] = FourierStackFine[two_nspace - i];

      /* Now do the FFT to get the a_M(theta) */
      fourier_trans_2(FourierStackCoarse-1,four_ntheta_m,1);
      AMThetaPtrP = AMTheta[ M] - ntheta_m;
      AMThetaPtrM = AMTheta[-M] - ntheta_m;
      for(i=0;i<=two_ntheta_m;i++) AMThetaPtrP[i] = AMThetaPtrM[i] = 0.;
      AMThetaPtrP[0] = FourierStackCoarse[0] / four_ntheta_m;

      /* the symmetry across theta => -theta is different for odd and even M.  The division by
       * four_ntheta_m is designed so that the forward and backward Fourier transforms will be
       * inverses of each other, without any multiplying factors.
       */
      if (M%2==0) { /* M even */
         for(i=1;i<two_ntheta_m;i++) {
            AMThetaPtrP[i] += FourierStackCoarse[eight_ntheta_m - 2*i   ] / four_ntheta_m;
            AMThetaPtrP[i] += FourierStackCoarse[2*i                    ] / four_ntheta_m;
            if (M!=0) {
               AMThetaPtrM[i] += FourierStackCoarse[eight_ntheta_m - 2*i +1] / four_ntheta_m;
               AMThetaPtrM[i] += FourierStackCoarse[2*i +1                 ] / four_ntheta_m;
            }
         }
         AMThetaPtrP[two_ntheta_m] = FourierStackCoarse[four_ntheta_m   ] / four_ntheta_m;
      } else { /* M odd */
         for(i=1;i<two_ntheta_m;i++) {
            AMThetaPtrP[i] -= FourierStackCoarse[eight_ntheta_m - 2*i   ] / four_ntheta_m;
            AMThetaPtrP[i] += FourierStackCoarse[2*i                    ] / four_ntheta_m;
            AMThetaPtrM[i] -= FourierStackCoarse[eight_ntheta_m - 2*i +1] / four_ntheta_m;
            AMThetaPtrM[i] += FourierStackCoarse[2*i +1                 ] / four_ntheta_m;
         }
      }

   } /* end M loop */

   /* Clean up memory */
   free_dmatrix(AMFine,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dvector(FourierStackFine,0,two_nspace-1);
   free_dvector(FourierStackCoarse,0,two_nspace_max-1);

   /* Compute the aLM's's */
   latitude_ana_1(ALM,AMTheta,dtheta,ntheta_m,lmax,convolve_kernel,convolve_flag);

   /* Clean up memory */
   free_dmatrix(AMTheta,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);

}

/* latitude_ana_vector_1
 * *** TRANSFORMS FROM VECTOR A_M(THETA) TO SPHERICAL HARMONIC A_LM ***
 *
 * This is the transpose function of latitude_syn_vector_1.
 *
 * This function performs the latitude part of a spherical harmonic transform.
 * The function is also capable of doing a convolution on the a_lm's after
 * doing the spherical harmonic transform.
 *
 * The storage conventions are:
 *
 * ALM_V, ALM_A = coefficients according to SPHERE_MODES->coefs
 * AMTheta[M][y] = Re a_M(pi/2 - y*dtheta) [M>=0]
 *                   Im a_|M|(pi/2 - y*dtheta) [M<0]
 *
 * func(theta,phi) = Re sum_{M>=0} a_M(theta) e^(iM phi)
 *
 * Dimensions: AMTheta_{v1,v2}[-lmax..lmax][-ntheta..ntheta]
 * (Note: we use this range but if ALM is defined outside of this,
 * this will not cause a problem.)
 *
 * If convolve_flag!=0, ALM_V[L][M] is multiplied by convolve_kernel_V[L] after
 * the transform, and similarly for ALM_A and convolve_kernel_A.
 */

void latitude_ana_vector_1(double **ALM_V, double **ALM_A, double **AMTheta_1,
   double **AMTheta_2, double dtheta, long ntheta, long lmax,
   double *convolve_kernel_V, double *convolve_kernel_A,
   unsigned short int convolve_flag) {

   long L,M,y,j;
   double *RecA, *RecB; /* recursion coefficients for generating Y_LM */
   double *RecAp, *RecBp;
   double *RecC;
   double *RecCp;
   double *Cmodes_V, *Smodes_V, *Cmodes_A, *Smodes_A;
   double *Cmp_V, *Smp_V, *Cmp_A, *Smp_A;
   double costheta, sintheta, Msintheta;
   double AbsWLM, AbsXLM;
   double *sinMGtheta;
   int *lambda_exp;
   int lambda_exp_current;
   double InitializerCoef;
   double normalization;
   double AbsYLp1M,AbsYLM,AbsYLm1M;
   double AMCEven_1, AMCOdd_1, AMSEven_1, AMSOdd_1;
   double AMCEven_2, AMCOdd_2, AMSEven_2, AMSOdd_2;
   unsigned short int latitude_parity;

   /* Allocate auxiliary arrays */
   RecA = dvector(0,lmax);
   RecB = dvector(0,lmax);
   RecC = dvector(0,lmax);
   Cmodes_V = dvector(0,lmax);
   Smodes_V = dvector(0,lmax);
   Cmodes_A = dvector(0,lmax);
   Smodes_A = dvector(0,lmax);
   sinMGtheta = dvector(0,ntheta);
   lambda_exp = ivector(0,ntheta);

   /* Initialize sin^(M-G)(theta) array and exponent overflow array */
   for(y=0;y<=ntheta;y++) {
      sinMGtheta[y] = 1.;
      lambda_exp[y] = 0;
   }

   /* Outer loop over the M-values */
   for(M=0;M<=lmax;M++) {

      /* Initialize recurrance relation coefficients */
      for(L=M;L<=lmax;L++) {

         /* Y recursions */
         RecA[L] = sqrt((2*L+3.0)/((L+1)*(L+1)-M*M));
         RecB[L] = sqrt((L*L-M*M)/(2*L-1.0))*RecA[L];
         RecA[L] *= sqrt(2*L+1.0);

         /* W and X recursions -- these depend on Y. */
         RecC[L] = (L>0)? sqrt( (2*L+1.0)/(2*L-1.0) * (L*L-M*M) ) : 0;
      }
      InitializerCoef=OneOverTwoPi;
      for(j=1;j<=M;j++) InitializerCoef *= 1.0 + 0.5/j;
      InitializerCoef = sqrt(InitializerCoef);
      if (M%2==1) InitializerCoef *= -1;

      /* Initialize harmonic modes for this value of M */
      for(L=M;L<=lmax;L++) {
         Cmodes_V[L] = Smodes_V[L] = 0;
         Cmodes_A[L] = Smodes_A[L] = 0;
      }

      /* Now run through all the points */
      for(y=0;y<=ntheta;y++) {
         /* Note that y*dtheta is the latitude but theta is the colatitude;
          * this is why sin and cos look backward but yes, this is correct.
          */
         costheta = sin(y*dtheta);
         sintheta = cos(y*dtheta);
         Msintheta = M * sintheta;

         /* Initialize the spherical harmonics at L=|M|; we will   
          * use the recurrance relation, based on NR 6.8.2,7 to
          * find the normalized latitude part of the spherical
          * harmonic.  The recurrance is:
          * sqrt((L^2-M^2)/(2L+1) * |Y_L_M| =
          *                        sqrt(2L-1) * cos(theta)|Y_L-1_M|
          *     - sqrt(((L-1)^2-M^2)/(2L-3)) * |Y_L-2_M|           
          * AbsYLMs are normalized to Int(AbsYLM sin^G theta)^2 dcostheta = 1/Pi
          */
         if (sinMGtheta[y] < LAMBDA_MIN) {
            sinMGtheta[y] *= INV_LAMBDA_MIN;
            lambda_exp[y]++;
         }
         lambda_exp_current = lambda_exp[y];
         AbsYLm1M = 0;
         AbsYLM = InitializerCoef * sinMGtheta[y];

         /* Increment sinMGtheta for next use */
         if (M>=1) sinMGtheta[y] *= sintheta;

         /* Initialize pointers */
         RecAp = RecA + M;
         RecBp = RecB + M;
         RecCp = RecC + M;
         Cmp_V = Cmodes_V + M;
         Smp_V = Smodes_V + M;
         Cmp_A = Cmodes_A + M;
         Smp_A = Smodes_A + M;

         /* Initialize the opposite-parity arrays and parity counter */
         AMCEven_1 = AMCOdd_1 = AMSEven_1 = AMSOdd_1 = 0.;
         AMCEven_2 = AMCOdd_2 = AMSEven_2 = AMSOdd_2 = 0.;
         latitude_parity = 0;

         /* This block of code will take the input arrays AMTheta_1, AMTheta_2
          * and put their values into the even/odd and cosine/sine coefficients,
          * which are used later (in the loop over L).  This step is unnecessary
          * (in fact, the code will crash!) if M<1 and sintheta==0, hence we
          * block that case.
          */
         if (M==1 || fabs(sintheta)>MIN_SIN_THETA) {

            /* Now put these results into the output arrays */
            AMCEven_1 = AMTheta_1[ M][ y] + AMTheta_1[ M][-y];
            AMCOdd_1  = AMTheta_1[ M][ y] - AMTheta_1[ M][-y];
            AMCEven_2 = AMTheta_2[ M][ y] + AMTheta_2[ M][-y];
            AMCOdd_2  = AMTheta_2[ M][ y] - AMTheta_2[ M][-y];
            if (M!=0) {
               AMSEven_1 = AMTheta_1[-M][ y] + AMTheta_1[-M][-y];
               AMSOdd_1  = AMTheta_1[-M][ y] - AMTheta_1[-M][-y];
               AMSEven_2 = AMTheta_2[-M][ y] + AMTheta_2[-M][-y];
               AMSOdd_2  = AMTheta_2[-M][ y] - AMTheta_2[-M][-y];
            }
            if (y==0) { /* We've double-counted the points on the equator. */
               AMCEven_1 /= 2.;
               AMSEven_1 /= 2.;
               AMCEven_2 /= 2.;
               AMSEven_2 /= 2.;
            }

            /* Divide the resulting v1 and v2 by sin(theta) if necessary */
            if (M==0) {
               AMCEven_1 /= sintheta;
               AMCEven_2 /= sintheta;
               AMCOdd_1  /= sintheta;
               AMCOdd_2  /= sintheta;
            }

         }

         /* Now go through the values of L */
         for(L=M;L<=lmax;L++) {

            if (lambda_exp_current>0) {
               /* If lambda_exp_current>0 we don't need to do the
                * multiplication/addition of the output, but we must
                * see whether Ylm is large enough that we can reduce
                * the number of orders of underflow, lambda_exp_current.
                */
               if (fabs(AbsYLM) > 1.0) {
                  lambda_exp_current--;
                  AbsYLM *= LAMBDA_MIN;
                  AbsYLm1M *= LAMBDA_MIN;
               }

            }

            if (lambda_exp_current==0) {

               /* First we need to compute W' and X'.  Note that AbsWLM and
                * AbsXLM are actually sin(theta) times the gradient of Ylm,
                * which we corrected for earlier.
                */
               AbsWLM = M * AbsYLM;
               AbsXLM = -L * costheta * AbsYLM + (*RecCp) * AbsYLm1M;

               /* Do the multiplication for each latitude ring
                * Note that we only increment the even or odd parity
                * a_m(theta)'s depending on the parity of the
                * particular Ylm.
                */
               if (latitude_parity) {
                  *Smp_V  -= AbsWLM * AMCOdd_1;
                  *Cmp_V  += AbsWLM * AMSOdd_1;
                  *Cmp_V  += AbsXLM * AMCEven_2;
                  *Smp_V  += AbsXLM * AMSEven_2;
                  *Smp_A  -= AbsWLM * AMCOdd_2;
                  *Cmp_A  += AbsWLM * AMSOdd_2;
                  *Cmp_A  -= AbsXLM * AMCEven_1;
                  *Smp_A  -= AbsXLM * AMSEven_1;
               } else {
                  *Smp_V  -= AbsWLM * AMCEven_1;
                  *Cmp_V  += AbsWLM * AMSEven_1;
                  *Cmp_V  += AbsXLM * AMCOdd_2;
                  *Smp_V  += AbsXLM * AMSOdd_2;
                  *Smp_A  -= AbsWLM * AMCEven_2;
                  *Cmp_A  += AbsWLM * AMSEven_2;
                  *Cmp_A  -= AbsXLM * AMCOdd_1;
                  *Smp_A  -= AbsXLM * AMSOdd_1;
               }

            }

            /* and then increment the Ylm and pointers */
            AbsYLp1M = (*RecAp) * costheta * AbsYLM - (*RecBp) * AbsYLm1M;
            RecAp++; RecBp++; RecCp++;
            Cmp_V++; Smp_V++; Cmp_A++; Smp_A++;
            AbsYLm1M = AbsYLM;
            AbsYLM = AbsYLp1M;

            latitude_parity ^= 1; /* switches between 0 and 1. */

         } /* end L loop */

      } /* end y loop */

      /* do the post-transform convolution, if necessary */
      if (convolve_flag) {
         for(L=M;L<=lmax;L++) {
            Cmodes_V[L] *= convolve_kernel_V[L];
            Smodes_V[L] *= convolve_kernel_V[L];
            Cmodes_A[L] *= convolve_kernel_A[L];
            Smodes_A[L] *= convolve_kernel_A[L];
         }
      }

      /* and now apply the normalization factor, i.e. the conversion
       * between (W+iX)_lm and (v1+iv2).
       */
      for(L=M;L<=lmax;L++) {
         normalization = (L>=1)? 1./sqrt((double)L*(L+1)) :0 ;
         Cmodes_V[L] *= normalization;
         Smodes_V[L] *= normalization;
         Cmodes_A[L] *= normalization;
         Smodes_A[L] *= normalization;
      }

      /* and construct the vectors of cosinelike and sinelike coefficients */
      if (M==0) for(L=0;L<=lmax;L++) {
         Smodes_V[L] = Smodes_A[L] = 0; /* no sin(0phi) mode */
         Cmodes_V[L] /= SqrtTwo;
         Cmodes_A[L] /= SqrtTwo;
      }
      for(L=M;L<=lmax;L++) {
         ALM_V[L][-M] += Smodes_V[L];
         ALM_A[L][-M] += Smodes_A[L];
         ALM_V[L][M]  += Cmodes_V[L];
         ALM_A[L][M]  += Cmodes_A[L];
      }

   } /* end M loop */

   /* De-allocate auxiliary arrays */
   free_dvector(RecA,0,lmax);
   free_dvector(RecB,0,lmax);
   free_dvector(RecC,0,lmax);
   free_dvector(Cmodes_V,0,lmax);
   free_dvector(Smodes_V,0,lmax);
   free_dvector(Cmodes_A,0,lmax);
   free_dvector(Smodes_A,0,lmax);
   free_dvector(sinMGtheta,0,ntheta);
   free_ivector(lambda_exp,0,ntheta);
}

/* sht_grid_analysis_vector_1
 * *** TRANSFORMS FROM VECTOR FUNC(THETA,PHI) TO SPHERICAL A_LM ON EQUIRECTANGULAR GRID ***
 *
 * The V_lm, A_lm must be in the SPHERE_MODES->coefs convention
 * The grid input is:
 *
 * {v1,v2}map[x][y] = {v1,v2}(phi = x*delta, theta = y*delta)
 *
 * where delta = 2pi/nspace
 * and the input range of func_map is [xmin..xmax][ymin..ymax].
 * (Note: we use this range but if func_map is defined outside of this,
 * this will not cause a problem.)
 *
 * nspace MUST be a multiple of 4.  It is best for it to be a power of two, or a
 * power of two times a small odd number (this is for the FFT).  If nspace is 
 * a multiple of 4 with lots of odd factors, this routine will still work but will
 * be slow.
 *
 * Note that the output is ADDED TO A_LM, NOT over-written.  If you want the latter you must
 * initialize A_LM before calling this routine.
 *
 * If convolve_flag!=0, the harmonic coefficients for ALM_V[L][M] is multiplied by convolve_kernel_V[L] after
 * the transform, and similarly for ALM_A and convolve_kernel_A.
 */

void sht_grid_analysis_vector_1(double **V_LM, double **A_LM, double **v1map, double **v2map, long lmax, long nspace,
   long xmin, long xmax, long ymin, long ymax, double *convolve_kernel_V, double *convolve_kernel_A,
   unsigned short int convolve_flag) {

   int vsign;
   long nspace_coarse, nspace_max, quarter_nspace_max, two_nspace_max;
   double dtheta;
   double **FuncThetaPhi, **func_map;
   double **AMTheta_1, **AMTheta_2, **AMTheta;
   double **AMFine_1, **AMFine_2, **AMFine;
   double *AMFinePtrM, *AMFinePtrP;
   double *AMThetaPtrM, *AMThetaPtrP;
   double *FourierStackFine, *FourierStackCoarse;
   long two_nspace, half_nspace, quarter_nspace;
   long ntheta_m, two_ntheta_m, four_ntheta_m, eight_ntheta_m;
   long M,i,x,y,j;
   long xoffset, yoffset;
   int k;

   /* Check that we were given a multiple of 4 */
   if (nspace % 4) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld should be a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Compute useful quantities related to nspace */
   two_nspace = nspace << 1;
   half_nspace = nspace >> 1;
   quarter_nspace = nspace >> 2;

   /* Determine how finely we need to sample theta.  Here dtheta is the spacing of our points in theta and
    * ntheta is the total number spaced around the entire great circle meridian (2pi).
    * We also determine nspace_max and quarter_nspace_max; nspace_max will be the maximum
    * number of points we need to consider on the entire meridian, quarter_nspace_max will
    * be only for the quarter-meridian from 0 to pi/2.
    */
   nspace_max = 0;
   nspace_coarse = optimum_length_1( 2*lmax+1, BETA_FFT_SHT);
      /* The "+1" in the optimum_length_1 argument is there so that we do not have
       * any power at the Nyquist frequency itself (this would result in aliasing
       * and consequent loss of information).
       */
   ntheta_m = nspace_coarse >> 2;
   two_ntheta_m = ntheta_m << 1;
   four_ntheta_m = ntheta_m << 2;
   eight_ntheta_m = ntheta_m << 3;
   dtheta = TwoPi/nspace_coarse;
   nspace_max = nspace_coarse;
   quarter_nspace_max = nspace_max >> 2;
   two_nspace_max = nspace_max << 1;

   /* Check that we are over-sampling */
   if (nspace_coarse > nspace) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld, lmax=%ld, this is not oversampled.\n",nspace,lmax);
      exit(1);
   }

   /* Allocate and compute the a_M(theta)'s */
   AMFine_1 = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   AMFine_2 = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   FourierStackFine = dvector(0,two_nspace-1);
   FourierStackCoarse = dvector(0,two_nspace_max-1);

   /* This is where we begin doing the sht_grid_synthesis_1 operations in reverse order.
    * We begin by allocating some memory for the longitude transform.
    */
   FuncThetaPhi = dmatrix(0,nspace-1,-quarter_nspace,quarter_nspace);

   /* Loop over {v1,v2} */
   for(k=0;k<2;k++) {
      AMFine = (k==0)? AMFine_1:AMFine_2;
      func_map = (k==0)? v1map:v2map;

      /* Clear FuncThetaPhi and AMFine */
      for(i=0;i<nspace;i++) {
         for(j=-quarter_nspace;j<=quarter_nspace;j++) {
            FuncThetaPhi[i][j] = 0.;
      }}
      for(i= -lmax;i<=lmax;i++) {
         for(j=-quarter_nspace;j<=quarter_nspace;j++) {
            AMFine[i][j] = 0.;
      }}

      /* Read the output map func_map into FuncThetaPhi */
      xoffset = (xmin>=0)? 0 : nspace * ((-xmin)/nspace + 1);
      yoffset = (ymin>=0)? 0 : nspace * ((-ymin)/nspace + 1);
      for(x=xmin;x<=xmax;x++) {
         for(y=ymin;y<=ymax;y++) {
            vsign = 1;
            i = (x + xoffset) % nspace;
            j = (y + yoffset) % nspace;
            if (j>=3*quarter_nspace) j -= nspace;
            if (j>quarter_nspace) {
               i = (i + half_nspace) % nspace;
               j = half_nspace - j;
               vsign = -1;            /* Flips sign because vector components are odd under
                                       * theta --> -theta and phi --> phi+pi
                                       */
            }

         FuncThetaPhi[i][j] += func_map[x][y] * vsign;
         }
      }

      /* For each j, we do the longitude FFT */
      for(j=-quarter_nspace;j<=quarter_nspace;j++) {

         /* load elements of AMFine into FourierStack */
         for(i=0;i<two_nspace;i++) FourierStackFine[i]=0;
         for(i=0;i<nspace;i++) FourierStackFine[2*i] = FuncThetaPhi[i][j];

         /* FFT from A(phi)(theta) to AM(theta) -- notice that we use reverse FFT,
          * i.e. the isign=-1 option is used here.  (We want to use the Hermitian
          * conjugate of the FFT matrix, this is the reverse FFT.)
          */
         fourier_trans_2(FourierStackFine-1,nspace,-1);

         AMFine[0][j] = FourierStackFine[0];
         for(M=1;M<=lmax;M++) {
            AMFine[ M][j] = FourierStackFine[2*M];
            AMFine[-M][j] = FourierStackFine[2*M+1];
         }

      } /* end j loop */
   } /* end k={v1,v2} loop */

   /* Reallocate some memory */
   free_dmatrix(FuncThetaPhi,0,nspace-1,-quarter_nspace,quarter_nspace);
   AMTheta_1 = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   AMTheta_2 = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);

   /* Loop over v1 and v2 for trig interpolation stage */
   for(k=0;k<2;k++) {
      AMTheta = (k==0)? AMTheta_1:AMTheta_2;
      AMFine = (k==0)? AMFine_1:AMFine_2;

      /* For each M, we do an exact trigonometric interpolation to yield
       * the values of am(theta) at a grid of points spaced in theta by
       * TwoPi/nspace_coarse.  The method is to consider a complete meridian
       * starting at the S pole, going up to the N pole, and then back
       * down at the opposite (+180deg) longitude.
       */
      for(M=0;M<=lmax;M++) {

         /* We will use FourierStackFine to do the reverse (transpose) FFT to take a_M(theta) at the
          * finely sampled points, and put this information on the coarse-grid.  Begin by clearing
          * it, then copying over positive, then negative frequency modes, finally do the FFT itself.
          */

         /* Put the AMFine values into the Fourier stack to be transformed */
         AMFinePtrP = AMFine[ M] - quarter_nspace;
         AMFinePtrM = AMFine[-M] - quarter_nspace;
         for(i=0;i<two_nspace;i++) FourierStackFine[i] = 0;
         for(i=0;i<=half_nspace;i++) {
            FourierStackFine[2*i  ] = AMFinePtrP[i];
            if (M!=0) FourierStackFine[2*i+1] = AMFinePtrM[i];
         }

         /* Transpose FFT, then copy the relevant Fourier modes over to FourierStackCoarse */
         fourier_trans_2(FourierStackFine-1,nspace,-1);
         for(i=0;i<two_nspace_max;i++) FourierStackCoarse[i] = 0;
         for(i=0;i<four_ntheta_m;i++) FourierStackCoarse[i] = FourierStackFine[i];
         for(i=1;i<=four_ntheta_m;i++) FourierStackCoarse[eight_ntheta_m - i] = FourierStackFine[two_nspace - i];

         /* Now do the FFT to get the a_M(theta) */
         fourier_trans_2(FourierStackCoarse-1,four_ntheta_m,1);
         AMThetaPtrP = AMTheta[ M] - ntheta_m;
         AMThetaPtrM = AMTheta[-M] - ntheta_m;
         for(i=0;i<=two_ntheta_m;i++) AMThetaPtrP[i] = AMThetaPtrM[i] = 0.;
         AMThetaPtrP[0] = FourierStackCoarse[0] / four_ntheta_m;
         AMThetaPtrM[0] = FourierStackCoarse[1] / four_ntheta_m;

         /* the symmetry across theta => -theta is different for odd and even M.  The division by
          * four_ntheta_m is designed so that the forward and backward Fourier transforms will be
          * inverses of each other, without any multiplying factors.
          *
          * Note that the parity of the vector is opposite that of scalar or tensor (it has odd
          * spin), so the symmetry under theta => -theta has an extra negative sign, compare
          * to sht_grid_analysis_1 and sht_grid_analysis_polar_1.
          */
         if (M%2==1) { /* M odd */
            for(i=1;i<two_ntheta_m;i++) {
               AMThetaPtrP[i] += FourierStackCoarse[eight_ntheta_m - 2*i   ] / four_ntheta_m;
               AMThetaPtrP[i] += FourierStackCoarse[2*i                    ] / four_ntheta_m;
               AMThetaPtrM[i] += FourierStackCoarse[eight_ntheta_m - 2*i +1] / four_ntheta_m;
               AMThetaPtrM[i] += FourierStackCoarse[2*i +1                 ] / four_ntheta_m;
            }
            AMThetaPtrP[two_ntheta_m] = FourierStackCoarse[four_ntheta_m   ] / four_ntheta_m;
            AMThetaPtrM[two_ntheta_m] = FourierStackCoarse[four_ntheta_m +1] / four_ntheta_m;
         } else { /* M even */
            for(i=1;i<two_ntheta_m;i++) {
               AMThetaPtrP[i] -= FourierStackCoarse[eight_ntheta_m - 2*i   ] / four_ntheta_m;
               AMThetaPtrP[i] += FourierStackCoarse[2*i                    ] / four_ntheta_m;
               AMThetaPtrM[i] -= FourierStackCoarse[eight_ntheta_m - 2*i +1] / four_ntheta_m;
               AMThetaPtrM[i] += FourierStackCoarse[2*i +1                 ] / four_ntheta_m;
            }
         }

      } /* end M loop */
   } /* end k={v1,v2} loop */

   /* Clean up memory */
   free_dmatrix(AMFine_1,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dmatrix(AMFine_2,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dvector(FourierStackFine,0,two_nspace-1);
   free_dvector(FourierStackCoarse,0,two_nspace_max-1);

   /* Compute the VLM's and ALM's */
   latitude_ana_vector_1(V_LM,A_LM,AMTheta_1,AMTheta_2,dtheta,ntheta_m,lmax,convolve_kernel_V,
      convolve_kernel_A,convolve_flag);

   /* Clean up memory */
   free_dmatrix(AMTheta_1,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   free_dmatrix(AMTheta_2,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);

}

/* latitude_ana_polar_1
 * *** TRANSFORMS FROM POLARIZATION A_M(THETA) TO SPHERICAL HARMONIC A_LM ***
 *
 * This is the transpose function of latitude_syn_polar_1.
 *
 * This function performs the latitude part of a spherical harmonic transform.
 * The function is also capable of doing a convolution on the a_lm's after
 * doing the spherical harmonic transform.
 *
 * The storage conventions are:
 *
 * ALM_E, ALM_B = coefficients according to SPHERE_MODES->coefs
 * AMTheta[M][y] = Re a_M(pi/2 - y*dtheta) [M>=0]
 *                   Im a_|M|(pi/2 - y*dtheta) [M<0]
 *
 * func(theta,phi) = Re sum_{M>=0} a_M(theta) e^(iM phi)
 *
 * Dimensions: AMTheta_{Q,U}[-lmax..lmax][-ntheta..ntheta]
 * (Note: we use this range but if ALM is defined outside of this,
 * this will not cause a problem.)
 *
 * If convolve_flag!=0, ALM_E[L][M] is multiplied by convolve_kernel_E[L] after
 * the transform, and similarly for ALM_B and convolve_kernel_B.
 */

void latitude_ana_polar_1(double **ALM_E, double **ALM_B, double **AMTheta_Q,
   double **AMTheta_U, double dtheta, long ntheta, long lmax,
   double *convolve_kernel_E, double *convolve_kernel_B,
   unsigned short int convolve_flag) {

   long L,M,y,j;
   double *RecA, *RecB; /* recursion coefficients for generating Y_LM */
   double *RecAp, *RecBp;
   double *RecC, *RecD, *RecE;
   double *RecCp, *RecDp, *RecEp;
   double *Cmodes_E, *Smodes_E, *Cmodes_B, *Smodes_B;
   double *Cmp_E, *Smp_E, *Cmp_B, *Smp_B;
   double costheta, sintheta;
   double AbsWLM, AbsXLM, sin2theta, multAbsYLm1M;
   double *sinMGtheta;
   int *lambda_exp;
   int lambda_exp_current;
   double InitializerCoef;
   double normalization;
   double AbsYLp1M,AbsYLM,AbsYLm1M;
   double AMCEven_Q, AMCOdd_Q, AMSEven_Q, AMSOdd_Q;
   double AMCEven_U, AMCOdd_U, AMSEven_U, AMSOdd_U;
   unsigned short int latitude_parity;

   /* Allocate auxiliary arrays */
   RecA = dvector(0,lmax);
   RecB = dvector(0,lmax);
   RecC = dvector(0,lmax);
   RecD = dvector(0,lmax);
   RecE = dvector(0,lmax);
   Cmodes_E = dvector(0,lmax);
   Smodes_E = dvector(0,lmax);
   Cmodes_B = dvector(0,lmax);
   Smodes_B = dvector(0,lmax);
   sinMGtheta = dvector(0,ntheta);
   lambda_exp = ivector(0,ntheta);

   /* Initialize sin^(M-G)(theta) array and exponent overflow array */
   for(y=0;y<=ntheta;y++) {
      sinMGtheta[y] = 1.;
      lambda_exp[y] = 0;
   }

   /* Outer loop over the M-values */
   for(M=0;M<=lmax;M++) {

      /* Initialize recurrance relation coefficients */
      for(L=M;L<=lmax;L++) {

         /* Y recursions */
         RecA[L] = sqrt((2*L+3.0)/((L+1)*(L+1)-M*M));
         RecB[L] = sqrt((L*L-M*M)/(2*L-1.0))*RecA[L];
         RecA[L] *= sqrt(2*L+1.0);

         /* W and X recursions -- these depend on Y. */
         RecC[L] = (double)(L-M*M);
         RecD[L] = 0.5*(L*(L-1));
         RecE[L] = (L>0)? sqrt( (L*L-M*M) * ((double)(2*L+1)/(2*L-1)) ) : 0;
      }
      InitializerCoef=OneOverTwoPi;
      for(j=1;j<=M;j++) InitializerCoef *= 1.0 + 0.5/j;
      InitializerCoef = sqrt(InitializerCoef);
      if (M%2==1) InitializerCoef *= -1;

      /* Initialize harmonic modes for this value of M */
      for(L=M;L<=lmax;L++) {
         Cmodes_E[L] = Smodes_E[L] = 0;
         Cmodes_B[L] = Smodes_B[L] = 0;
      }

      /* Now run through all the points */
      for(y=0;y<=ntheta;y++) {
         /* Note that y*dtheta is the latitude but theta is the colatitude;
          * this is why sin and cos look backward but yes, this is correct.
          */
         costheta = sin(y*dtheta);
         sintheta = cos(y*dtheta);
         sin2theta = sintheta * sintheta;

         /* Initialize the spherical harmonics at L=|M|; we will   
          * use the recurrance relation, based on NR 6.8.2,7 to
          * find the normalized latitude part of the spherical
          * harmonic.  The recurrance is:
          * sqrt((L^2-M^2)/(2L+1) * |Y_L_M| =
          *                        sqrt(2L-1) * cos(theta)|Y_L-1_M|
          *     - sqrt(((L-1)^2-M^2)/(2L-3)) * |Y_L-2_M|           
          * AbsYLMs are normalized to Int(AbsYLM sin^G theta)^2 dcostheta = 1/Pi
          */
         if (sinMGtheta[y] < LAMBDA_MIN) {
            sinMGtheta[y] *= INV_LAMBDA_MIN;
            lambda_exp[y]++;
         }
         lambda_exp_current = lambda_exp[y];
         AbsYLm1M = 0;
         AbsYLM = InitializerCoef * sinMGtheta[y];

         /* Increment sinMGtheta for next use */
         if (M>=2) sinMGtheta[y] *= sintheta;

         /* Initialize pointers */
         RecAp = RecA + M;
         RecBp = RecB + M;
         RecCp = RecC + M;
         RecDp = RecD + M;
         RecEp = RecE + M;
         Cmp_E = Cmodes_E + M;
         Smp_E = Smodes_E + M;
         Cmp_B = Cmodes_B + M;
         Smp_B = Smodes_B + M;

         /* Initialize the opposite-parity arrays and parity counter */
         AMCEven_Q = AMCOdd_Q = AMSEven_Q = AMSOdd_Q = 0.;
         AMCEven_U = AMCOdd_U = AMSEven_U = AMSOdd_U = 0.;
         latitude_parity = 0;

         /* This block of code will take the input arrays AMTheta_Q, AMTheta_U
          * and put their values into the even/odd and cosine/sine coefficients,
          * which are used later (in the loop over L).  This step is unnecessary
          * (in fact, the code will crash!) if M<2 and sintheta==0, hence we
          * block that case.
          */
         if (M==2 || fabs(sintheta)>MIN_SIN_THETA) {

            /* Now put these results into the output arrays */
            AMCEven_Q = AMTheta_Q[ M][ y] + AMTheta_Q[ M][-y];
            AMCOdd_Q  = AMTheta_Q[ M][ y] - AMTheta_Q[ M][-y];
            AMCEven_U = AMTheta_U[ M][ y] + AMTheta_U[ M][-y];
            AMCOdd_U  = AMTheta_U[ M][ y] - AMTheta_U[ M][-y];
            if (M!=0) {
               AMSEven_Q = AMTheta_Q[-M][ y] + AMTheta_Q[-M][-y];
               AMSOdd_Q  = AMTheta_Q[-M][ y] - AMTheta_Q[-M][-y];
               AMSEven_U = AMTheta_U[-M][ y] + AMTheta_U[-M][-y];
               AMSOdd_U  = AMTheta_U[-M][ y] - AMTheta_U[-M][-y];
            }
            if (y==0) { /* We've double-counted the points on the equator. */
               AMCEven_Q /= 2.;
               AMSEven_Q /= 2.;
               AMCEven_U /= 2.;
               AMSEven_U /= 2.;
            }

            /* Divide the resulting Q and U by sin^(2-G)(theta) if necessary */
            if (M==0) {
               AMCEven_Q /= sin2theta;
               AMCEven_U /= sin2theta;
               AMCOdd_Q  /= sin2theta;
               AMCOdd_U  /= sin2theta;
            }
            if (M==1) {
               AMCEven_Q /= sintheta;
               AMCEven_U /= sintheta;
               AMCOdd_Q  /= sintheta;
               AMCOdd_U  /= sintheta;
               AMSEven_Q /= sintheta;
               AMSEven_U /= sintheta;
               AMSOdd_Q  /= sintheta;
               AMSOdd_U  /= sintheta;
            }

         }

         /* Now go through the values of L */
         for(L=M;L<=lmax;L++) {

            if (lambda_exp_current>0) {
               /* If lambda_exp_current>0 we don't need to do the
                * multiplication/addition of the output, but we must
                * see whether Ylm is large enough that we can reduce
                * the number of orders of underflow, lambda_exp_current.
                */
               if (fabs(AbsYLM) > 1.0) {
                  lambda_exp_current--;
                  AbsYLM *= LAMBDA_MIN;
                  AbsYLm1M *= LAMBDA_MIN;
               }

            }

            if (lambda_exp_current==0) {

               /* First we need to compute W and X.  Note that AbsWLM and
                * AbsXLM are actually (sin^2theta)/2 times W and X.  The factor
                * of 2 is accounted for in the definitions of YG and YC, whereas
                * the factor of sin^(2-G)(theta) is inserted after the sum.
                */
               multAbsYLm1M = (*RecEp) * AbsYLm1M;
               AbsWLM = (*RecCp + (*RecDp) * sin2theta) * AbsYLM - costheta * multAbsYLm1M;
               AbsXLM = M*( (L-1) * costheta * AbsYLM - multAbsYLm1M );

               /* Do the multiplication for each latitude ring
                * Note that we only increment the even or odd parity
                * a_m(theta)'s depending on the parity of the
                * particular Ylm.
                */
               if (latitude_parity) {
                  *Cmp_E  += AbsWLM * AMCOdd_Q;
                  *Smp_E  += AbsWLM * AMSOdd_Q;
                  *Cmp_B  += AbsXLM * AMSEven_Q;
                  *Smp_B  -= AbsXLM * AMCEven_Q;
                  *Cmp_B  += AbsWLM * AMCOdd_U;
                  *Smp_B  += AbsWLM * AMSOdd_U;
                  *Cmp_E  -= AbsXLM * AMSEven_U;
                  *Smp_E  += AbsXLM * AMCEven_U;
               } else {
                  *Cmp_E  += AbsWLM * AMCEven_Q;
                  *Smp_E  += AbsWLM * AMSEven_Q;
                  *Cmp_B  += AbsXLM * AMSOdd_Q;
                  *Smp_B  -= AbsXLM * AMCOdd_Q;
                  *Cmp_B  += AbsWLM * AMCEven_U;
                  *Smp_B  += AbsWLM * AMSEven_U;
                  *Cmp_E  -= AbsXLM * AMSOdd_U;
                  *Smp_E  += AbsXLM * AMCOdd_U;
               }

            }

            /* and then increment the Ylm and pointers */
            AbsYLp1M = (*RecAp) * costheta * AbsYLM - (*RecBp) * AbsYLm1M;
            RecAp++; RecBp++; RecCp++; RecDp++; RecEp++;
            Cmp_E++; Smp_E++; Cmp_B++; Smp_B++;
            AbsYLm1M = AbsYLM;
            AbsYLM = AbsYLp1M;

            latitude_parity ^= 1; /* switches between 0 and 1. */

         } /* end L loop */

      } /* end y loop */

      /* do the post-transform convolution, if necessary */
      if (convolve_flag) {
         for(L=M;L<=lmax;L++) {
            Cmodes_E[L] *= convolve_kernel_E[L];
            Smodes_E[L] *= convolve_kernel_E[L];
            Cmodes_B[L] *= convolve_kernel_B[L];
            Smodes_B[L] *= convolve_kernel_B[L];
         }
      }

      /* and now apply the normalization factor, i.e. the conversion
       * between (W+iX)_lm and (Q+iU).  Note that we are using the EB
       * convention, as distinct from the GC convention, hence our
       * normalization factor is different from Kamionkowski et al
       * N_l (see astro-ph/9611125v1 eq 2.16) by a factor of sqrt2.
       * Notice that we wipe out modes with L<2.
       */
      for(L=M;L<=lmax;L++) {
         normalization = (L>=2)? 2./sqrt(((double)(L-1)*L)*(L+1)*(L+2)) :0 ;
         Cmodes_E[L] *= normalization;
         Smodes_E[L] *= normalization;
         Cmodes_B[L] *= normalization;
         Smodes_B[L] *= normalization;
      }

      /* and construct the vectors of cosinelike and sinelike coefficients */
      if (M==0) for(L=0;L<=lmax;L++) {
         Smodes_E[L] = Smodes_B[L] = 0; /* no sin(0phi) mode */
         Cmodes_E[L] /= SqrtTwo;
         Cmodes_B[L] /= SqrtTwo;
      }
      for(L=M;L<=lmax;L++) {
         ALM_E[L][-M] += Smodes_E[L];
         ALM_B[L][-M] += Smodes_B[L];
         ALM_E[L][M]  += Cmodes_E[L];
         ALM_B[L][M]  += Cmodes_B[L];
      }

   } /* end M loop */

   /* De-allocate auxiliary arrays */
   free_dvector(RecA,0,lmax);
   free_dvector(RecB,0,lmax);
   free_dvector(RecC,0,lmax);
   free_dvector(RecD,0,lmax);
   free_dvector(RecE,0,lmax);
   free_dvector(Cmodes_E,0,lmax);
   free_dvector(Smodes_E,0,lmax);
   free_dvector(Cmodes_B,0,lmax);
   free_dvector(Smodes_B,0,lmax);
   free_dvector(sinMGtheta,0,ntheta);
   free_ivector(lambda_exp,0,ntheta);
}

/* sht_grid_analysis_polar_1
 * *** TRANSFORMS FROM POLARIZATION FUNC(THETA,PHI) TO SPHERICAL A_LM ON EQUIRECTANGULAR GRID ***
 *
 * The E_lm, B_lm must be in the SPHERE_MODES->coefs convention
 * The grid output is:
 *
 * {Q,U}map[x][y] = {Q,U}(phi = x*delta, theta = y*delta)
 *
 * where delta = 2pi/nspace
 * and the output range of func_map is [xmin..xmax][ymin..ymax].
 * (Note: we use this range but if func_map is defined outside of this,
 * this will not cause a problem.)
 *
 * nspace MUST be a multiple of 4.  It is best for it to be a power of two, or a
 * power of two times a small odd number (this is for the FFT).  If nspace is 
 * a multiple of 4 with lots of odd factors, this routine will still work but will
 * be slow.
 *
 * Note that the output is ADDED TO A_LM, NOT over-written.  If you want the latter you must
 * initialize A_LM before calling this routine.
 *
 * If convolve_flag!=0, the harmonic coefficents for ALM_E[L][M] is multiplied by convolve_kernel_E[L] after
 * the transform, and similarly for ALM_B and convolve_kernel_B.
 */

void sht_grid_analysis_polar_1(double **E_LM, double **B_LM, double **Qmap, double **Umap, long lmax, long nspace,
   long xmin, long xmax, long ymin, long ymax, double *convolve_kernel_E, double *convolve_kernel_B,
   unsigned short int convolve_flag) {

   long nspace_coarse, nspace_max, quarter_nspace_max, two_nspace_max;
   double dtheta;
   double **FuncThetaPhi, **func_map;
   double **AMTheta_Q, **AMTheta_U, **AMTheta;
   double **AMFine_Q, **AMFine_U, **AMFine;
   double *AMFinePtrM, *AMFinePtrP;
   double *AMThetaPtrM, *AMThetaPtrP;
   double *FourierStackFine, *FourierStackCoarse;
   long two_nspace, half_nspace, quarter_nspace;
   long ntheta_m, two_ntheta_m, four_ntheta_m, eight_ntheta_m;
   long M,i,x,y,j;
   long xoffset, yoffset;
   int k;

   /* Check that we were given a multiple of 4 */
   if (nspace % 4) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld should be a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Compute useful quantities related to nspace */
   two_nspace = nspace << 1;
   half_nspace = nspace >> 1;
   quarter_nspace = nspace >> 2;

   /* Determine how finely we need to sample theta.  Here dtheta is the spacing of our points in theta and
    * ntheta is the total number spaced around the entire great circle meridian (2pi).
    * We also determine nspace_max and quarter_nspace_max; nspace_max will be the maximum
    * number of points we need to consider on the entire meridian, quarter_nspace_max will
    * be only for the quarter-meridian from 0 to pi/2.
    */
   nspace_max = 0;
   nspace_coarse = optimum_length_1( 2*lmax+1, BETA_FFT_SHT);
      /* The "+1" in the optimum_length_1 argument is there so that we do not have
       * any power at the Nyquist frequency itself (this would result in aliasing
       * and consequent loss of information).
       */
   ntheta_m = nspace_coarse >> 2;
   two_ntheta_m = ntheta_m << 1;
   four_ntheta_m = ntheta_m << 2;
   eight_ntheta_m = ntheta_m << 3;
   dtheta = TwoPi/nspace_coarse;
   nspace_max = nspace_coarse;
   quarter_nspace_max = nspace_max >> 2;
   two_nspace_max = nspace_max << 1;

   /* Check that we are over-sampling */
   if (nspace_coarse > nspace) {
      fprintf(stderr,"Error in sht_grid_synthesis_1: nspace=%ld, lmax=%ld, this is not oversampled.\n",nspace,lmax);
      exit(1);
   }

   /* Allocate and compute the a_M(theta)'s */
   AMFine_Q = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   AMFine_U = dmatrix(-lmax,lmax,-quarter_nspace,quarter_nspace);
   FourierStackFine = dvector(0,two_nspace-1);
   FourierStackCoarse = dvector(0,two_nspace_max-1);

   /* This is where we begin doing the sht_grid_synthesis_1 operations in reverse order.
    * We begin by allocating some memory for the longitude transform.
    */
   FuncThetaPhi = dmatrix(0,nspace-1,-quarter_nspace,quarter_nspace);

   /* Loop over {Q,U} */
   for(k=0;k<2;k++) {
      AMFine = (k==0)? AMFine_Q:AMFine_U;
      func_map = (k==0)? Qmap:Umap;

      /* Clear FuncThetaPhi and AMFine */
      for(i=0;i<nspace;i++) {
         for(j=-quarter_nspace;j<=quarter_nspace;j++) {
            FuncThetaPhi[i][j] = 0.;
      }}
      for(i= -lmax;i<=lmax;i++) {
         for(j=-quarter_nspace;j<=quarter_nspace;j++) {
            AMFine[i][j] = 0.;
      }}

      /* Read the output map func_map into FuncThetaPhi */
      xoffset = (xmin>=0)? 0 : nspace * ((-xmin)/nspace + 1);
      yoffset = (ymin>=0)? 0 : nspace * ((-ymin)/nspace + 1);
      for(x=xmin;x<=xmax;x++) {
         for(y=ymin;y<=ymax;y++) {
            i = (x + xoffset) % nspace;
            j = (y + yoffset) % nspace;
            if (j>=3*quarter_nspace) j -= nspace;
            if (j>quarter_nspace) {
               i = (i + half_nspace) % nspace;
               j = half_nspace - j;
            }

         FuncThetaPhi[i][j] += func_map[x][y];
         }
      }

      /* For each j, we do the longitude FFT */
      for(j=-quarter_nspace;j<=quarter_nspace;j++) {

         /* load elements of AMFine into FourierStack */
         for(i=0;i<two_nspace;i++) FourierStackFine[i]=0;
         for(i=0;i<nspace;i++) FourierStackFine[2*i] = FuncThetaPhi[i][j];

         /* FFT from A(phi)(theta) to AM(theta) -- notice that we use reverse FFT,
          * i.e. the isign=-1 option is used here.  (We want to use the Hermitian
          * conjugate of the FFT matrix, this is the reverse FFT.)
          */
         fourier_trans_2(FourierStackFine-1,nspace,-1);

         AMFine[0][j] = FourierStackFine[0];
         for(M=1;M<=lmax;M++) {
            AMFine[ M][j] = FourierStackFine[2*M];
            AMFine[-M][j] = FourierStackFine[2*M+1];
         }

      } /* end j loop */
   } /* end k={Q,U} loop */

   /* Reallocate some memory */
   free_dmatrix(FuncThetaPhi,0,nspace-1,-quarter_nspace,quarter_nspace);
   AMTheta_Q = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   AMTheta_U = dmatrix(-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);

   /* Loop over Q and U for trig interpolation stage */
   for(k=0;k<2;k++) {
      AMTheta = (k==0)? AMTheta_Q:AMTheta_U;
      AMFine = (k==0)? AMFine_Q:AMFine_U;

      /* For each M, we do an exact trigonometric interpolation to yield
       * the values of am(theta) at a grid of points spaced in theta by
       * TwoPi/nspace_coarse.  The method is to consider a complete meridian
       * starting at the S pole, going up to the N pole, and then back
       * down at the opposite (+180deg) longitude.
       */
      for(M=0;M<=lmax;M++) {

         /* We will use FourierStackFine to do the reverse (transpose) FFT to take a_M(theta) at the
          * finely sampled points, and put this information on the coarse-grid.  Begin by clearing
          * it, then copying over positive, then negative frequency modes, finally do the FFT itself.
          */

         /* Put the AMFine values into the Fourier stack to be transformed */
         AMFinePtrP = AMFine[ M] - quarter_nspace;
         AMFinePtrM = AMFine[-M] - quarter_nspace;
         for(i=0;i<two_nspace;i++) FourierStackFine[i] = 0;
         for(i=0;i<=half_nspace;i++) {
            FourierStackFine[2*i  ] = AMFinePtrP[i];
            if (M!=0) FourierStackFine[2*i+1] = AMFinePtrM[i];
         }

         /* Transpose FFT, then copy the relevant Fourier modes over to FourierStackCoarse */
         fourier_trans_2(FourierStackFine-1,nspace,-1);
         for(i=0;i<two_nspace_max;i++) FourierStackCoarse[i] = 0;
         for(i=0;i<four_ntheta_m;i++) FourierStackCoarse[i] = FourierStackFine[i];
         for(i=1;i<=four_ntheta_m;i++) FourierStackCoarse[eight_ntheta_m - i] = FourierStackFine[two_nspace - i];

         /* Now do the FFT to get the a_M(theta) */
         fourier_trans_2(FourierStackCoarse-1,four_ntheta_m,1);
         AMThetaPtrP = AMTheta[ M] - ntheta_m;
         AMThetaPtrM = AMTheta[-M] - ntheta_m;
         for(i=0;i<=two_ntheta_m;i++) AMThetaPtrP[i] = AMThetaPtrM[i] = 0.;
         AMThetaPtrP[0] = FourierStackCoarse[0] / four_ntheta_m;
         AMThetaPtrM[0] = FourierStackCoarse[1] / four_ntheta_m;

         /* the symmetry across theta => -theta is different for odd and even M.  The division by
          * four_ntheta_m is designed so that the forward and backward Fourier transforms will be
          * inverses of each other, without any multiplying factors.
          */
         if (M%2==0) { /* M even */
            for(i=1;i<two_ntheta_m;i++) {
               AMThetaPtrP[i] += FourierStackCoarse[eight_ntheta_m - 2*i   ] / four_ntheta_m;
               AMThetaPtrP[i] += FourierStackCoarse[2*i                    ] / four_ntheta_m;
               AMThetaPtrM[i] += FourierStackCoarse[eight_ntheta_m - 2*i +1] / four_ntheta_m;
               AMThetaPtrM[i] += FourierStackCoarse[2*i +1                 ] / four_ntheta_m;
            }
            AMThetaPtrP[two_ntheta_m] = FourierStackCoarse[four_ntheta_m   ] / four_ntheta_m;
            AMThetaPtrM[two_ntheta_m] = FourierStackCoarse[four_ntheta_m +1] / four_ntheta_m;
         } else { /* M odd */
            for(i=1;i<two_ntheta_m;i++) {
               AMThetaPtrP[i] -= FourierStackCoarse[eight_ntheta_m - 2*i   ] / four_ntheta_m;
               AMThetaPtrP[i] += FourierStackCoarse[2*i                    ] / four_ntheta_m;
               AMThetaPtrM[i] -= FourierStackCoarse[eight_ntheta_m - 2*i +1] / four_ntheta_m;
               AMThetaPtrM[i] += FourierStackCoarse[2*i +1                 ] / four_ntheta_m;
            }
         }

      } /* end M loop */
   } /* end k={Q,U} loop */

   /* Clean up memory */
   free_dmatrix(AMFine_Q,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dmatrix(AMFine_U,-lmax,lmax,-quarter_nspace, quarter_nspace);
   free_dvector(FourierStackFine,0,two_nspace-1);
   free_dvector(FourierStackCoarse,0,two_nspace_max-1);

   /* Compute the ELM's and BLM's */
   latitude_ana_polar_1(E_LM,B_LM,AMTheta_Q,AMTheta_U,dtheta,ntheta_m,lmax,convolve_kernel_E,
      convolve_kernel_B,convolve_flag);

   /* Clean up memory */
   free_dmatrix(AMTheta_Q,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);
   free_dmatrix(AMTheta_U,-lmax,lmax,-quarter_nspace_max,quarter_nspace_max);

}

/* sphere_step_1
 * *** TAKES A STEP OF GIVEN HEADING AND LENGTH ON SPHERE, RETURNS COORDINATES AND ROTATION ANGLE ***
 *
 * Starts at position (theta0,phi0) on sphere, then takes a step in direction heading with length
 * lstep (in radians).  The heading is zero for a step that is due "east" (phi increasing),
 * pi/2 for a step that is due "north" (theta decreasing), pi for a step that is due "west"
 * (phi decreasing), and 3pi/2 for a step that is due "south" (theta increasing).  The
 * function places the new (theta,phi) in the appropriate locations.  It returns psi, the angle
 * of rotation of the coordinate axes if we parallel-transport along the great circle arc
 * connecting (theta,phi) and (theta0,phi0).  Specifically, if (Vx,Vy) at (theta,phi) is a vector
 * that is parallel-transported to (Vx0,Vy0) at (theta0,phi0), we have:
 *
 * Vx0 = Vx cos psi - Vy sin psi
 * Vy0 = Vx sin psi + Vy cos psi
 *
 * The phases become 2*psi if you are transporting a tensor with Q,U components.
 */
double sphere_step_1(double theta0, double phi0, double lstep, double heading, double *theta, double *phi) {

   double costheta;

   /* Find colatitude */
   costheta = cos(theta0)*cos(lstep) + sin(theta0)*sin(lstep)*sin(heading);
   if (fabs(costheta)<1) {
      *theta = acos(costheta);

      /* Use spherical triangle for the longitude */
      *phi = atan2( sin(lstep)*cos(heading), sin(theta0)*cos(lstep) - cos(theta0)*sin(lstep)*sin(heading) ) + phi0;
      if (*phi>TwoPi) *phi -= TwoPi;
      if (*phi<0) *phi += TwoPi;
      return ( heading - atan2( -cos(theta0)*sin(lstep) + sin(theta0)*cos(lstep)*sin(heading), sin(theta0)*cos(heading) ) );

   };

   /* If we get here, we are in the polar regions */
   *theta = (costheta>0)? 0: Pi;
   *phi = phi0;
   return(0);

}

/* sphere_coord_bound_1
 * *** MOVES (THETA,PHI) COORDINATES INTO STANDARD RANGE ***
 *
 * The standard range is: 0<=theta<=pi, 0<=phi<2*pi.
 *
 */
void sphere_coord_bound_1(double *theta, double *phi) {

   unsigned int rev;

   /* Make theta positive if necessary */
   if (*theta < 0) {
      *theta = - *theta;
      *phi += Pi;
   }

   /* Now put theta in 0..TwoPi */
   rev = (unsigned int) floor (*theta/TwoPi);
   *theta -= TwoPi*rev;

   /* And now in 0..Pi */
   if (*theta > Pi) {
      *theta = TwoPi - *theta;
      *phi += Pi;
   }

   /* Put phi in 0..TwoPi */
   if (*phi >= TwoPi) {
      rev = (unsigned int) floor (*phi/TwoPi);
      *phi -= TwoPi*rev;
   }
   if (*phi < 0) {
      rev = (unsigned int) floor (1-*phi/TwoPi);
      *phi += TwoPi*rev;
   }

}

/* forward_sphere_pm_1
 * *** FORWARD INTERPOLATION ON THE SPHERE ***
 *
 * Forward-interpolation (i.e. mesh=>particle) on a spherical map.
 *
 * Arguments:
 *   func: the function to be interpolated
 *   theta: the colatitude vector of the particle positions
 *   phi: the longitude vector of the particle positions
 * > f: the outputs to which we add func(theta,phi)
 *   M: the order of polynomial interpolation used
 *   length: the number of particles for which the function is to be computed
 *   quarter_nspace: the grid resolution to be used.
 */

void forward_sphere_pm_1(double **func, double *theta, double *phi, double *f,
   int M, long length, long quarter_nspace) {

   long i;
   double *x, *y;
   double angle_space = PiOverTwo / quarter_nspace;

   /* Find the xy-coordinates of the particles */
   x = dvector(0,length-1);
   y = dvector(0,length-1);
   for(i=0;i<length;i++) {
      x[i] = phi[i] / angle_space;
      y[i] = quarter_nspace - theta[i] / angle_space;
   }

   forward_grid_interp_1(func,x,y,f,M,length);

   free_dvector(x,0,length-1);
   free_dvector(y,0,length-1);

}

/* reverse_sphere_pm_1
 * *** REVERSE INTERPOLATION ON THE SPHERE ***
 *
 * Reverse-interpolation (i.e. particle=>mesh) on a spherical map.
 *
 * Arguments:
 * > func: the mesh onto which we interpolate
 *   theta: the colatitude vector of the particle positions
 *   phi: the longitude vector of the particle positions
 *   f: the inputs at which we add func(theta,phi)
 *   M: the order of polynomial interpolation used
 *   length: the number of particles for which the function is to be computed
 *   quarter_nspace: the grid resolution to be used.
 */

void reverse_sphere_pm_1(double **func, double *theta, double *phi, double *f,
   int M, long length, long quarter_nspace) {

   long i;
   double *x, *y;
   double angle_space = PiOverTwo / quarter_nspace;

   /* Find the xy-coordinates of the particles */
   x = dvector(0,length-1);
   y = dvector(0,length-1);
   for(i=0;i<length;i++) {
      x[i] = phi[i] / angle_space;
      y[i] = quarter_nspace - theta[i] / angle_space;
   }

   reverse_grid_interp_1(func,x,y,f,M,length);

   free_dvector(x,0,length-1);
   free_dvector(y,0,length-1);
}

/* pixel_synthesis_1
 * *** SHT SYNTHESIS ON A SET OF PIXELS ***
 *
 * Performs harmonic => real space conversion on a set of pixels.  The result is added to the output map,
 * not over-written.
 *
 * Arguments:
 *   p_LM: + parity (S, V, or E) component of harmonic representation of the function
 *   n_LM: - parity (A or B) component of harmonic representation of the function
 *      (not used for scalar transform, can pass NULL)
 * > component_1: the real-space representation (S, X, or Q component)
 * > component_2: the real-space representation (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 *   lmax: maximum multipole used in the transform
 *   theta: colatitude vector of pixels
 *   phi: longitude vector of pixels
 *   psi: orientation vector of pixels (0 = X to East, Y to North; pi/2 = X to North, Y to West)
 *   length: number of pixels used for transform
 *   convolve_kernel_p: convolution kernel used for + parity components [0..lmax].  Used if convolve_flag==1,
 *      and not if convolve_flag==0.
 *   convolve_kernel_n: convolution kernel used for - parity components [0..lmax].  Used if convolve_flag==1,
 *      and not if convolve_flag==0.
 *   spin: type of transform: 0 = scalar, 1 = vector, 2 = tensor.
 *   convolve_flag: apply convolution kernel?  [Yes=1, No=0]
 *   nspace: size of grid used for interpolation
 *   pm_order: order of PM interpolation
 */

void pixel_synthesis_1(SPHERE_MODES *p_LM, SPHERE_MODES *n_LM, double *component_1, double *component_2,
   long lmax, double *theta, double *phi, double *psi, long length, double *convolve_kernel_p,
   double *convolve_kernel_n, int spin, unsigned short int convolve_flag, long nspace, long pm_order) {

   long i;
   long xmin,xmax,ymin,ymax;
   long quarter_nspace = nspace/4;
#ifndef PSIZERO
   double comp1,comp2,cos_spsi,sin_spsi;
#endif
   DOUBLE_MAP C1, C2;
   double *rot_component_1, *rot_component_2;

   rot_component_2 = NULL; /* This avoids the compiler initialization warning, although we will */
                           /* assign a value if spin!=0, i.e. if we actually use it.            */

   /* Check value of nspace */
   if (nspace%4) {
      fprintf(stderr,"Error in pixel_synthesis_1: nspace=%ld is not a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Allocate memory */
   allocate_double_map(&C1, xmin = -pm_order-1,
                            xmax = nspace+pm_order+1,
                            ymin = -quarter_nspace-pm_order-1,
                            ymax = quarter_nspace+pm_order+1, 0);
   if (spin) allocate_double_map(&C2,xmin,xmax,ymin,ymax,0);

   /* Do transform */
   switch (spin) {

      case 0: /* Scalar transform */
         sht_grid_synthesis_1(p_LM->coefs,C1.matrix,lmax,nspace,xmin,xmax,ymin,ymax,convolve_kernel_p,
            convolve_flag);
         break;

      case 1: /* Vector transform */
         sht_grid_synthesis_vector_1(p_LM->coefs,n_LM->coefs,C1.matrix,C2.matrix,lmax,nspace,
            xmin,xmax,ymin,ymax,convolve_kernel_p,convolve_kernel_n,convolve_flag);
         break;

      case 2: /* Tensor transform */
         sht_grid_synthesis_polar_1(p_LM->coefs,n_LM->coefs,C1.matrix,C2.matrix,lmax,nspace,
            xmin,xmax,ymin,ymax,convolve_kernel_p,convolve_kernel_n,convolve_flag);
         break;

      default: /* User did not specify S, V, or T, hence an error */
         fprintf(stderr,"Error in pixel_synthesis_1: spin=%d must be 0, 1, or 2.\n",spin);
         exit(1);
   }

   /* Now grid the data onto the given pixels */
   rot_component_1 = dvector(0,length-1);
   if (spin) rot_component_2 = dvector(0,length-1);
   for(i=0;i<length;i++) rot_component_1[i] = 0.;
   if (spin) for(i=0;i<length;i++) rot_component_2[i] = 0.;
   forward_sphere_pm_1(C1.matrix,theta,phi,rot_component_1,pm_order,length,quarter_nspace);
   if (spin) {
      forward_sphere_pm_1(C2.matrix,theta,phi,rot_component_2,pm_order,length,quarter_nspace);

      /* Vectors and tensors must be rotated */
      for(i=0;i<length;i++) {
#ifdef PSIZERO
         component_1[i] += rot_component_1[i];
         component_2[i] += rot_component_2[i];
#endif
#ifndef PSIZERO
         comp1 = rot_component_1[i];
         comp2 = rot_component_2[i];
         cos_spsi = cos(spin*psi[i]);
         sin_spsi = sin(spin*psi[i]);
         component_1[i] += comp1 * cos_spsi + comp2 * sin_spsi;
         component_2[i] += comp2 * cos_spsi - comp1 * sin_spsi;
#endif
      }
   } else {
      for(i=0;i<length;i++) component_1[i] += rot_component_1[i];
   }

   free_dvector(rot_component_1,0,length-1);
   if (spin) free_dvector(rot_component_2,0,length-1);

   deallocate_double_map(&C1);
   if (spin) deallocate_double_map(&C2);
}

/* pixel_analysis_1
 * *** SHT ANALYSIS ON A SET OF PIXELS ***
 *
 * Performs harmonic => real space conversion on a set of pixels.  The result is added to the output
 * map, not over-written.
 *
 * Arguments:
 *   p_LM: + parity (S, V, or E) component of harmonic representation of the function
 *   n_LM: - parity (A or B) component of harmonic representation of the function
 *      (not used for scalar transform, can pass NULL)
 * > component_1: the real-space representation (S, X, or Q component)
 * > component_2: the real-space representation (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 *   lmax: maximum multipole used in the transform
 *   theta: colatitude vector of pixels
 *   phi: longitude vector of pixels
 *   psi: orientation vector of pixels (0 = X to East, Y to North; pi/2 = X to North, Y to West)
 *   area: area vector of pixels (only used if convolve_flag 0x2 bit is on)
 *   length: number of pixels used for transform
 *   convolve_kernel_p: convolution kernel used for + parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   convolve_kernel_n: convolution kernel used for - parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   spin: type of transform: 0 = scalar, 1 = vector, 2 = tensor.
 *   convolve_flag: two bits: 0x1 = apply convolution kernel? [Yes=1, No=0]
 *                            0x2 = multiply by area? [Yes=1, No=0]
 *   nspace: size of grid used for interpolation
 *   pm_order: order of PM interpolation
 */

void pixel_analysis_1(SPHERE_MODES *p_LM, SPHERE_MODES *n_LM, double *component_1, double *component_2,
   long lmax, double *theta, double *phi, double *psi, double *area, long length, double *convolve_kernel_p,
   double *convolve_kernel_n, int spin, unsigned short int convolve_flag, long nspace, long pm_order) {

   long i;
   long xmin,xmax,ymin,ymax;
   long quarter_nspace = nspace/4;
#ifndef PSIZERO
   double cos_spsi,sin_spsi;
#endif
   DOUBLE_MAP C1, C2;
   double *rot_component_1, *rot_component_2;

   rot_component_2 = NULL; /* This avoids the compiler initialization warning, although we will */
                           /* assign a value if spin!=0, i.e. if we actually use it.            */

   /* Check value of nspace */
   if (nspace%4) {
      fprintf(stderr,"Error in pixel_analysis_1: nspace=%ld is not a multiple of 4.\n",nspace);
      exit(1);
   }

   /* Rotate coordinates */
   rot_component_1 = dvector(0,length-1);
   if (spin) rot_component_2 = dvector(0,length-1);
   for(i=0;i<length;i++) {
      if (spin) { /* Must rotate the pixel */
#ifdef PSIZERO
         rot_component_1[i] = component_1[i];
         rot_component_2[i] = component_2[i];
#endif
#ifndef PSIZERO
         cos_spsi = cos(spin*psi[i]);
         sin_spsi = sin(spin*psi[i]);
         rot_component_1[i] = component_1[i] * cos_spsi - component_2[i] * sin_spsi;
         rot_component_2[i] = component_2[i] * cos_spsi + component_1[i] * sin_spsi;
#endif
      } else {
         rot_component_1[i] = component_1[i];
      }
      if (convolve_flag & 0x2) {
         rot_component_1[i] *= area[i];
         if (spin) rot_component_2[i] *= area[i];
      }
   }

   /* Allocate memory for grid */
   allocate_double_map(&C1, xmin = -pm_order-1,
                           xmax = nspace+pm_order+1,
                           ymin = -quarter_nspace-pm_order-1,
                           ymax = quarter_nspace+pm_order+1, 0);
   if (spin) allocate_double_map(&C2,xmin,xmax,ymin,ymax,0);

   /* Make grid */
   reverse_sphere_pm_1(C1.matrix,theta,phi,rot_component_1,pm_order,length,quarter_nspace);
   if (spin) reverse_sphere_pm_1(C2.matrix,theta,phi,rot_component_2,pm_order,length,quarter_nspace);

   /* Clean up memory for rotated components */
   free_dvector(rot_component_1,0,length-1);
   if (spin) free_dvector(rot_component_2,0,length-1);

   /* Do transform */
   switch (spin) {

      case 0: /* Scalar transform */
         sht_grid_analysis_1(p_LM->coefs,C1.matrix,lmax,nspace,xmin,xmax,ymin,ymax,convolve_kernel_p,
            convolve_flag & 0x1);
         break;

      case 1: /* Vector transform */
         sht_grid_analysis_vector_1(p_LM->coefs,n_LM->coefs,C1.matrix,C2.matrix,lmax,nspace,
            xmin,xmax,ymin,ymax,convolve_kernel_p,convolve_kernel_n,convolve_flag & 0x1);
         break;

      case 2: /* Tensor transform */
         sht_grid_analysis_polar_1(p_LM->coefs,n_LM->coefs,C1.matrix,C2.matrix,lmax,nspace,
            xmin,xmax,ymin,ymax,convolve_kernel_p,convolve_kernel_n,convolve_flag & 0x1);
         break;

      default: /* User did not specify S, V, or T, hence an error */
         fprintf(stderr,"Error in pixel_analysis_1: spin=%d must be 0, 1, or 2.\n",spin);
         exit(1);
   }

   deallocate_double_map(&C1);
   if (spin) deallocate_double_map(&C2);
}

/* pixel_convolution_1
 * *** SPHERICAL CONVOLUTION ON A SET OF PIXELS ***
 *
 * Performs real space spherical convolution on a set of pixels.  The result is added to the output map,
 * not over-written.
 *
 * Arguments:
 *   component_1_in: the input map (S, X, or Q component)
 *   component_2_in: the input map (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 * > component_1_out: the output map (S, X, or Q component)
 * > component_2_out: the output map (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 *   lmax: maximum multipole used in the convolution
 *   theta_in: colatitude vector of pixels -- input map
 *   phi_in: longitude vector of pixels -- input map
 *   psi_in: orientation vector of pixels (0 = X to East, Y to North; pi/2 = X to North, Y to West) -- input map
 *   area_in: area vector of pixels (only used if convolve_flag 0x2 bit is on) -- input map
 *   npix_in: number of pixels -- input map
 *   theta_out: colatitude vector of pixels -- output map
 *   phi_out: longitude vector of pixels -- output map
 *   psi_out: orientation vector of pixels (0 = X to East, Y to North; pi/2 = X to North, Y to West) -- output map
 *   npix_out: number of pixels -- output map
 *   convolve_kernel_p: convolution kernel used for + parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   convolve_kernel_n: convolution kernel used for - parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   spin_in: type of transform: 0 = scalar, 1 = vector, 2 = tensor. (input)
 *   spin_out: type of transform: 0 = scalar, 1 = vector, 2 = tensor. (output)
 *   convolve_flag: two bits: 0x1 = apply convolution kernel? [Yes=1, No=0]
 *                            0x2 = multiply by area? [Yes=1, No=0]
 *   nspace: size of grid used for interpolation
 *   pm_order: order of PM interpolation
 */

void pixel_convolution_1(double *component_1_in, double *component_2_in, double *component_1_out, double *component_2_out,
   long lmax, double *theta_in, double *phi_in, double *psi_in, double *area_in, long npix_in, double *theta_out,
   double *phi_out, double *psi_out, long npix_out, double *convolve_kernel_p, double *convolve_kernel_n,
   int spin_in, int spin_out, unsigned short int convolve_flag, long nspace, long pm_order) {

   SPHERE_MODES p_LM;
   SPHERE_MODES n_LM;

   allocate_sphere_modes(&p_LM,lmax);
   if (spin_in || spin_out) allocate_sphere_modes(&n_LM,lmax);

   pixel_analysis_1(&p_LM,&n_LM,component_1_in,component_2_in,lmax,theta_in,phi_in,psi_in,area_in,npix_in,convolve_kernel_p,
      convolve_kernel_n,spin_in,convolve_flag,nspace,pm_order);

   pixel_synthesis_1(&p_LM,&n_LM,component_1_out,component_2_out,lmax,theta_out,phi_out,psi_out,npix_out,NULL,NULL,
      spin_out,0,nspace,pm_order);

   deallocate_sphere_modes(&p_LM);
   if (spin_in || spin_out) deallocate_sphere_modes(&n_LM);
}

/* pixel_convolution_2
 * *** SPHERICAL CONVOLUTION ON A SET OF PIXELS ***
 *
 * Performs real space spherical convolution on a set of pixels.  The result is added to the output map,
 * not over-written.
 *
 * Arguments:
 *   component_1_in: the input map (S, X, or Q component)
 *   component_2_in: the input map (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 * > component_1_out: the output map (S, X, or Q component)
 * > component_2_out: the output map (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 *   lmax: maximum multipole used in the convolution
 *   in_pix: input pixelization of the sphere
 *   out_pix: output pixelization of the sphere
 *   convolve_kernel_p: convolution kernel used for + parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   convolve_kernel_n: convolution kernel used for - parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   spin_in: type of transform: 0 = scalar, 1 = vector, 2 = tensor. (input)
 *   spin_out: type of transform: 0 = scalar, 1 = vector, 2 = tensor. (output)
 *   convolve_flag: two bits: 0x1 = apply convolution kernel? [Yes=1, No=0]
 *                            0x2 = multiply by area? [Yes=1, No=0]
 *   nspace: size of grid used for interpolation
 *   pm_order: order of PM interpolation
 */

void pixel_convolution_2(double *component_1_in, double *component_2_in, double *component_1_out, double *component_2_out,
   long lmax, SPHERE_PIXEL *in_pix, SPHERE_PIXEL *out_pix, double *convolve_kernel_p, double *convolve_kernel_n,
   int spin_in, int spin_out, unsigned short int convolve_flag, long nspace, long pm_order) {

   SPHERE_MODES p_LM;
   SPHERE_MODES n_LM;

   allocate_sphere_modes(&p_LM,lmax);
   if (spin_in || spin_out) allocate_sphere_modes(&n_LM,lmax);

#ifdef N_CHECKVAL
   if ((convolve_flag&0x2) && (in_pix->area_flag==0)) {
      fprintf(stderr,"Error in pixel_convolution_2: input area undefined.\n");
      exit(1);
   }
#endif

   pixel_analysis_1(&p_LM,&n_LM,component_1_in,component_2_in,lmax,in_pix->theta,in_pix->phi,in_pix->psi,in_pix->area,in_pix->N,
      convolve_kernel_p,convolve_kernel_n,spin_in,convolve_flag,nspace,pm_order);

   pixel_synthesis_1(&p_LM,&n_LM,component_1_out,component_2_out,lmax,out_pix->theta,out_pix->phi,out_pix->psi,out_pix->N,
      NULL,NULL,spin_out,0,nspace,pm_order);

   deallocate_sphere_modes(&p_LM);
   if (spin_in || spin_out) deallocate_sphere_modes(&n_LM);
}

/* make_healpix_ring
 * *** ALLOCATES A HEALPIX-RING SPHERICAL PIXELIZATION ***
 *
 * Generates a HEALPix ring-format pixelization.
 *
 * Arguments:
 * > A: pointer to SPHERE_PIXEL structure that will hold pixelization
 *   resolution: resolution number (nside = 2**resolution)
 *   area_flag: allocate pixel areas? (1=yes, 0=no)
 */

void make_healpix_ring(SPHERE_PIXEL *A, int resolution, unsigned short area_flag) {

   long nside = 1<<resolution;
   double pix_area;
   long i,y,Nphi,x;
   double SinLatitude, InitPhi, StepPhi;

   /* Avoid initialization warnings -- we will always assign values before we use
    * these, but the compiler doesn't know that because it doesn't know that we
    * always go through one of the if (y...) blocks.
    */
   SinLatitude = InitPhi = StepPhi = 0.0; Nphi = 0;
  
   allocate_sphere_pixel(A, 12*nside*nside, area_flag);

   /* Run through pixels */
   i=0;
   for(y=1;y<=4*nside-1;y++) {
      /* find sin(latitude), Nphi (# pts on a ring), and */
      /* InitPhi,StepPhi (longitudes) */
      if (y<=nside) {
         SinLatitude = 1. - y*y/((double)(3*nside*nside));
         Nphi = 4*y;
         InitPhi = Pi/Nphi;
         StepPhi = TwoPi/Nphi;
      }
      if (y>nside && y<3*nside) {
         SinLatitude = 2./((double)(3*nside)) * (2*nside-y);
         Nphi = 4*nside;
         InitPhi = (y&1)? 0: Pi/Nphi;
         StepPhi = TwoPi/Nphi;
      }
      if (y>=3*nside) {
         SinLatitude = -1. + (4*nside-y)*(4*nside-y)/((double)(3*nside*nside));
         Nphi = 4*(4*nside-y);
         InitPhi = Pi/Nphi;
         StepPhi = TwoPi/Nphi;
      }
      for(x=0;x<Nphi;x++) {
         A->theta[i] = acos(SinLatitude);
         A->phi[i] = InitPhi + StepPhi * x;
         A->psi[i] = 0.;
         i++;
      }
   }

   /* Report pixel areas */
   if (area_flag) {
      pix_area = Pi / (3*nside*nside);
      for(i=0;i<A->N;i++) A->area[i] = pix_area;
   }
}

/* woodbury_preconditioner_1
 * *** PRECONDITIONER FOR C^{-1}X OPERATION ***
 *
 * This preconditioner is designed for use with very red C matrices
 * (i.e. the lowest few multipoles have very large eigenvalue that
 * dominates the condition ratio).  This version only works with scalars.
 *
 * The Woodbury formula is:
 *
 * (N + U V')^-1 = N^-1 - N^-1 U (1 + V' N^-1 U)^-1 V' N^-1
 *
 * where here we will set N = white noise = Cl[lsplit0],
 * U = V = (sht synthesis matrix) * sqrt(Cl[L]-Cl[lsplit0]),
 *
 * Thus we have an exact inversion C^-1 up to multipole lsplit0, which
 * is hopefully good enough to remove any ill conditioning.
 *
 * The OpSelect codes are:
 *   0: do an approximate C^{-1}X
 *   1: recompute transfer matrix for given power spectrum Cl and lsplit
 *   2: write transfer matrix to a file
 *   3: read transfer matrix from a file
 *   4: clean up memory
 *
 * Arguments:
 *   X: vector on which to do C^-1 operation
 * > ApproxCInvX: preconditioner * X
 *   theta: colatitudes of pixels
 *   phi: longitudes of pixels
 *   psi: pixel orientations (not actually used for scalars)
 *   area: solid angles of pixels
 *   npix: number of pixels
 *   Cl: power spectrum to precondition
 *   lsplit: maximum L to use for direct-inversion
 *   FileName: name of file to read/write
 *   OpSelect: which operation? (see above)
 *   tnum: which transfer matrix? ( 0 .. TNUM_MAX )
 */
#ifndef THREAD_SAFE_SHT
void woodbury_preconditioner_1(double *X, double *ApproxCInvX,
   double *theta, double *phi, double *psi, double *area, long npix,
   double *Cl, long lsplit, char FileName[], int OpSelect, int tnum) {

   FILE *fp;
   double *YLMRealSpace, *XIntermediate;
   double *WhiteNoiseLevel;
   double **sqrt_Cl_prior;
   double ***TransferMatrix;
   long *lsplit0;
   long *Nmodes;
   long i,j, L, M;
   long *MultipoleNum;
   SPHERE_MODES YLMIn, YLMOut;
   static WOODBURY_TRANSFER wtMatrix[TNUM_MAX+1];
   long nspace_grid, nspace_short, pix;
   SPHERE_PIXEL WoodGrid;
   double dtheta;
   double *ModeMap;
   long nspace_conv;

   /* Set all the pointers at the appropriate wtMatrix */
   WhiteNoiseLevel  = &(wtMatrix[tnum].WhiteNoiseLevel);
   sqrt_Cl_prior    = &(wtMatrix[tnum].sqrt_Cl_prior  );
   TransferMatrix   = &(wtMatrix[tnum].TransferMatrix );
   lsplit0          = &(wtMatrix[tnum].lsplit0        );
   Nmodes           = &(wtMatrix[tnum].Nmodes         );

   /* Select grid size for the convolution */
   nspace_conv = 1;
   while (nspace_conv < lsplit) nspace_conv<<=1;
   nspace_conv *= 8;
   if (nspace_conv < 10*lsplit) nspace_conv = (3*nspace_conv)>>1;

#define LSPLIT_MAX 100

   /* Decide which operation to perform */
   switch (OpSelect) {

      case 4: /* Clean up memory */
         free_dmatrix(*TransferMatrix, 0, *Nmodes-1, 0, *Nmodes-1);
         free_dvector(*sqrt_Cl_prior, 0, *lsplit0-1);
         *Nmodes = 0;
         return;

      case 3: /* Read transfer matrix from file */
         fp = fopen(FileName, "r");
         fscanf(fp, "%ld", lsplit0);
#ifdef N_CHECKVAL
         if (*lsplit0 > LSPLIT_MAX || *lsplit0<0) {
            fprintf(stderr,"Error: lsplit0=%ld out of range.\n", *lsplit0);
            exit(1);
         }
#endif
         *Nmodes = *lsplit0 * *lsplit0;
         fscanf(fp,"%lg",WhiteNoiseLevel);
         *TransferMatrix = dmatrix(0,*Nmodes-1,0,*Nmodes-1);
         *sqrt_Cl_prior = dvector(0,*lsplit0);
         for(L=0;L<*lsplit0;L++)
            fscanf(fp,"%lg", *sqrt_Cl_prior+L);
         for(i=0;i<*Nmodes;i++)
            for(j=0;j<*Nmodes;j++)
               fscanf(fp, "%lg", (*TransferMatrix)[i]+j);
         fclose(fp);
         return;

      case 2: /* Write transfer matrix to file */
#ifdef N_CHECKVAL
         if (!(*Nmodes)) {
            fprintf(stderr,"Error: no transfer matrix.\n");
            exit(1);
         }
#endif
         fp = fopen(FileName, "w");
         fprintf(fp,"%ld\n%23.16lE\n", *lsplit0, *WhiteNoiseLevel);
         for(L=0;L<*lsplit0;L++)
            fprintf(fp,"%23.16lE\n", (*sqrt_Cl_prior)[L]);
         for(i=0;i<*Nmodes;i++)
            for(j=0;j<*Nmodes;j++)
               fprintf(fp,"%23.16lE\n", (*TransferMatrix)[i][j]);
         fclose(fp);
         return;

      case 1: /* Compute the transfer matrix */

         /* Save user's value of lsplit, allocate memory */
         *lsplit0 = lsplit;
#ifdef N_CHECKVAL
         if (*lsplit0 > LSPLIT_MAX || *lsplit0<0) {
            fprintf(stderr,"Error: lsplit0=%ld out of range.\n", *lsplit0);
            exit(1);
         }
#endif
         *Nmodes = *lsplit0 * *lsplit0;
         allocate_sphere_modes(&YLMIn , *lsplit0-1);
         allocate_sphere_modes(&YLMOut, *lsplit0-1);
         *sqrt_Cl_prior = dvector(0,*lsplit0-1);
         YLMRealSpace = dvector(0,npix-1);
         MultipoleNum = lvector(0, *Nmodes-1);
         *TransferMatrix = dmatrix(0,*Nmodes-1,0,*Nmodes-1);
#ifdef N_CHECKVAL
         if ((YLMIn.Nmode != *Nmodes) || (YLMOut.Nmode != *Nmodes)) {
            fprintf(stderr, "Failure in woodbury_preconditioner_1.\n");
            exit(1);
         }
#endif

         /* Compute the prior power spectrum and effective white noise level */
         *WhiteNoiseLevel = Cl[*lsplit0];
         for(L=0;L<*lsplit0;L++) (*sqrt_Cl_prior)[L] = Cl[L]>*WhiteNoiseLevel? sqrt(Cl[L]-*WhiteNoiseLevel): 0.;

         /* Interpolate the pixelization onto a new one called WoodGrid, an ECL grid with
          * spacing of dtheta=2*Pi/(nspace_grid).
          */
         nspace_grid = 1;
         while (nspace_grid <= *lsplit0 + 2) nspace_grid<<=1;
         nspace_grid = 4*nspace_grid + 16;
         nspace_short = nspace_grid>>1;
         allocate_sphere_pixel(&WoodGrid, nspace_grid*nspace_short, 1);
         ModeMap = dvector(0, WoodGrid.N-1);
         dtheta = TwoPi / (double) nspace_grid;
         for(i=0;i<nspace_grid;i++) {
            for(j=0;j<nspace_short;j++) {
               pix = i*nspace_short+j;
               WoodGrid.theta[pix] = dtheta*(j+0.5);
               WoodGrid.phi  [pix] = dtheta*i;
               WoodGrid.psi  [pix] = 0.0;
               WoodGrid.area [pix] = 0.0;
            }
         }
         pixel_convolution_1(area, NULL, WoodGrid.area, NULL, (*lsplit0<<1)+4, theta, phi, psi, NULL, npix, WoodGrid.theta,
            WoodGrid.phi, WoodGrid.psi, WoodGrid.N, NULL, NULL, 0, 0, 0, nspace_conv, 5);
         for(pix=0;pix<WoodGrid.N;pix++) WoodGrid.area[pix] *= sin(WoodGrid.theta[pix]) * dtheta * dtheta;

         /* Which multipole number corresponds to which mode in the vector */
         for(L=0;L<*lsplit0;L++)
            for(M=-L;M<=L;M++)
               MultipoleNum[L*(L+1)+M] = L;

         /* The transfer matrix is computed one column at a time */
         for(i=0;i<*Nmodes;i++) {

            /* Build a unit column vector for use in the preconditioner.  The WhiteNoiseLevel
             * in the denominator provides the "N^-1" in the Woodbury formula.
             */
            for(j=0;j<*Nmodes;j++) YLMOut.vector[j] = YLMIn.vector[j] = 0.;
            YLMIn.vector[i] = (*sqrt_Cl_prior)[L = MultipoleNum[i]] / (*WhiteNoiseLevel);

            /* Now do spherical convolution on set of pixels.  Note that we
             * are going from harmonic-->real-->harmonic space.  We will use
             * pm_order = 3.
             */
            for(j=0;j<WoodGrid.N;j++) ModeMap[j] = 0.;
            pixel_synthesis_1(&YLMIn, NULL, ModeMap, NULL, *lsplit0-1, WoodGrid.theta, WoodGrid.phi, WoodGrid.psi,
               WoodGrid.N, NULL, NULL, 0, 0, nspace_conv, 3);
            for(j=0;j<WoodGrid.N;j++) ModeMap[j] *= WoodGrid.area[j];
            pixel_analysis_1(&YLMOut, NULL, ModeMap, NULL, *lsplit0-1, WoodGrid.theta, WoodGrid.phi, WoodGrid.psi,
               NULL, WoodGrid.N, *sqrt_Cl_prior, NULL, 0, 1, nspace_conv, 3);

            /* Store convolution results in transfer matrix */
            for(j=0;j<*Nmodes;j++) (*TransferMatrix)[i][j] = YLMOut.vector[j];

         }

         /* Now we have computed V' N^-1 U.  We want to get (1 + V' N^-1 U)^-1, which
          * is obtained by adding the identity and inverting.
          */
         for(i=0;i<*Nmodes;i++) (*TransferMatrix)[i][i] += 1.;
         gaussjinv(*TransferMatrix,*Nmodes);

         /* We won't be needing these anymore so kill them */
         free_lvector(MultipoleNum,0,*Nmodes-1);
         free_dvector(ModeMap, 0, WoodGrid.N-1);
         deallocate_sphere_pixel(&WoodGrid);
         deallocate_sphere_modes(&YLMIn );
         deallocate_sphere_modes(&YLMOut);
         return;

      case 0: /* Actually do a C^{-1}X computation using Woodbury */

         /* Allocate memory for intermediate steps */
         allocate_sphere_modes(&YLMIn , *lsplit0-1);
         allocate_sphere_modes(&YLMOut, *lsplit0-1);
         XIntermediate = dvector(0,npix-1);

         /* Do Woodbury formula.  First step is to compute XIntermediate = N^-1 X,
          * then YLMIn = V' N^-1 X.
          */
         for(i=0;i<npix;i++) XIntermediate[i] = X[i] / (*WhiteNoiseLevel) * area[i];
         pixel_analysis_1(&YLMIn, NULL, XIntermediate, NULL, *lsplit0-1, theta, phi, psi, NULL, npix, *sqrt_Cl_prior,
            NULL, 0, 1, nspace_conv, 3);
         for(i=0;i<npix;i++) XIntermediate[i] = 0.;

         /* Multiply by transfer matrix to get YLMOut = (1 + V' N^-1 U)^-1 V' N^-1 X. */
         for(i=0;i<*Nmodes;i++) {
            YLMOut.vector[i] = 0.;
            for(j=0;j<*Nmodes;j++)
               YLMOut.vector[i] += (*TransferMatrix)[j][i] * YLMIn.vector[j];
            }

         /* Finish by computing ApproxCInvX = (N+UV')^-1 X. */
         pixel_synthesis_1(&YLMOut, NULL, XIntermediate, NULL, *lsplit0-1, theta, phi, psi, npix, *sqrt_Cl_prior, NULL,
            0, 1, nspace_conv, 3);
         for(i=0;i<npix;i++) ApproxCInvX[i] += (X[i] - XIntermediate[i]) / (*WhiteNoiseLevel) * area[i];

         /* Clean up memory */
         deallocate_sphere_modes(&YLMIn );
         deallocate_sphere_modes(&YLMOut);
         free_dvector(XIntermediate,0,npix-1);
         return;

      default: /* We'd better not end up here */
            fprintf(stderr, "Error: unknown OpSelect code %d", OpSelect);
            exit(1);
   }
}
#endif

/* woodbury_preconditioner_2
 * *** PRECONDITIONER FOR C^{-1}X OPERATION ***
 *
 * This preconditioner is designed for use with very red C matrices
 * (i.e. the lowest few multipoles have very large eigenvalue that
 * dominates the condition ratio).  This version only works with scalars.
 *
 * This version of Woodbury keeps only half of the matrix since it is symmetric.
 *
 * Arguments:
 *   X: vector on which to do C^-1 operation
 * > ApproxCInvX: preconditioner * X
 *   theta: colatitudes of pixels
 *   phi: longitudes of pixels
 *   psi: pixel orientations (not actually used for scalars)
 *   area: solid angles of pixels
 *   npix: number of pixels
 *   Cl: power spectrum to precondition
 *   lsplit: maximum L to use for direct-inversion
 *   FileName: name of file to read/write
 *   OpSelect: which operation? (see above)
 *   tnum: which transfer matrix? ( 0 .. TNUM_MAX )
 */
#ifndef THREAD_SAFE_SHT
void woodbury_preconditioner_2(double *X, double *ApproxCInvX,
   double *theta, double *phi, double *psi, double *area, long npix,
   double *Cl, long lsplit, char FileName[], int OpSelect, int tnum) {

   FILE *fp;
   double *YLMRealSpace, *XIntermediate;
   double *WhiteNoiseLevel;
   double **sqrt_Cl_prior;
   double ***TransferMatrix;
   long *lsplit0;
   long *Nmodes;
   long i,j, k, L, M;
   long *MultipoleNum;
   SPHERE_MODES YLMIn, YLMOut;
   static WOODBURY_TRANSFER wtMatrix[TNUM_MAX+1];
   long nspace_grid, nspace_short, pix;
   SPHERE_PIXEL WoodGrid;
   double dtheta;
   double *ModeMap;
   long nspace_conv;
   double *ep;

   /* Set all the pointers at the appropriate wtMatrix */
   WhiteNoiseLevel  = &(wtMatrix[tnum].WhiteNoiseLevel);
   sqrt_Cl_prior    = &(wtMatrix[tnum].sqrt_Cl_prior  );
   TransferMatrix   = &(wtMatrix[tnum].TransferMatrix );
   lsplit0          = &(wtMatrix[tnum].lsplit0        );
   Nmodes           = &(wtMatrix[tnum].Nmodes         );

   /* Select grid size for the convolution */
   nspace_conv = 1;
   while (nspace_conv < lsplit) nspace_conv<<=1;
   nspace_conv *= 8;
   if (nspace_conv < 10*lsplit) nspace_conv = (3*nspace_conv)>>1;

#define LSPLIT_MAX 100

   /* Decide which operation to perform */
   switch (OpSelect) {

      case 4: /* Clean up memory */
         free_dvector(wtMatrix[tnum].element, 0, wtMatrix[tnum].nsparse-1);
         wtMatrix[tnum].element = 0;
         free_dvector(*sqrt_Cl_prior, 0, *lsplit0-1);
         *Nmodes = 0;
         return;

      case 3: /* Read transfer matrix from file */
         fp = fopen(FileName, "r");
         fscanf(fp, "%ld", lsplit0);
#ifdef N_CHECKVAL
         if (*lsplit0 > LSPLIT_MAX || *lsplit0<0) {
            fprintf(stderr,"Error: lsplit0=%ld out of range.\n", *lsplit0);
            exit(1);
         }
#endif
         *Nmodes = *lsplit0 * *lsplit0;
         fscanf(fp,"%ld",&(wtMatrix[tnum].nsparse));
         fscanf(fp,"%lg",WhiteNoiseLevel);
         wtMatrix[tnum].element = dvector(0, wtMatrix[tnum].nsparse-1);
         *sqrt_Cl_prior = dvector(0,*lsplit0);
         for(L=0;L<*lsplit0;L++)
            fscanf(fp,"%lg", *sqrt_Cl_prior+L);
         for(i=0; i<wtMatrix[tnum].nsparse; i++) fscanf(fp, "%lg", wtMatrix[tnum].element+i);
         fclose(fp);
         return;

      case 2: /* Write transfer matrix to file */
#ifdef N_CHECKVAL
         if (!(*Nmodes)) {
            fprintf(stderr,"Error: no transfer matrix.\n");
            exit(1);
         }
#endif
         fp = fopen(FileName, "w");
         fprintf(fp,"%ld %ld %23.16lE\n", *lsplit0, wtMatrix[tnum].nsparse, *WhiteNoiseLevel);
         for(L=0;L<*lsplit0;L++)
            fprintf(fp,"%23.16lE\n", (*sqrt_Cl_prior)[L]);
         for(i=0; i<wtMatrix[tnum].nsparse; i++) fprintf(fp,"%23.16lE\n", wtMatrix[tnum].element[i]);
         fclose(fp);
         return;

      case 1: /* Compute the transfer matrix */

         /* Save user's value of lsplit, allocate memory */
         *lsplit0 = lsplit;
#ifdef N_CHECKVAL
         if (*lsplit0 > LSPLIT_MAX || *lsplit0<0) {
            fprintf(stderr,"Error: lsplit0=%ld out of range.\n", *lsplit0);
            exit(1);
         }
#endif
         *Nmodes = *lsplit0 * *lsplit0;
         allocate_sphere_modes(&YLMIn , *lsplit0-1);
         allocate_sphere_modes(&YLMOut, *lsplit0-1);
         *sqrt_Cl_prior = dvector(0,*lsplit0-1);
         YLMRealSpace = dvector(0,npix-1);
         MultipoleNum = lvector(0, *Nmodes-1);
         *TransferMatrix = dmatrix(0,*Nmodes-1,0,*Nmodes-1);
#ifdef N_CHECKVAL
         if ((YLMIn.Nmode != *Nmodes) || (YLMOut.Nmode != *Nmodes)) {
            fprintf(stderr, "Failure in woodbury_preconditioner_1.\n");
            exit(1);
         }
#endif

         /* Compute the prior power spectrum and effective white noise level */
         *WhiteNoiseLevel = Cl[*lsplit0];
         for(L=0;L<*lsplit0;L++) (*sqrt_Cl_prior)[L] = Cl[L]>*WhiteNoiseLevel? sqrt(Cl[L]-*WhiteNoiseLevel): 0.;

         /* Interpolate the pixelization onto a new one called WoodGrid, an ECL grid with
          * spacing of dtheta=2*Pi/(nspace_grid).
          */
         nspace_grid = 1;
         while (nspace_grid <= *lsplit0 + 2) nspace_grid<<=1;
         nspace_grid = 4*nspace_grid + 16;
         nspace_short = nspace_grid>>1;
         allocate_sphere_pixel(&WoodGrid, nspace_grid*nspace_short, 1);
         ModeMap = dvector(0, WoodGrid.N-1);
         dtheta = TwoPi / (double) nspace_grid;
         for(i=0;i<nspace_grid;i++) {
            for(j=0;j<nspace_short;j++) {
               pix = i*nspace_short+j;
               WoodGrid.theta[pix] = dtheta*(j+0.5);
               WoodGrid.phi  [pix] = dtheta*i;
               WoodGrid.psi  [pix] = 0.0;
               WoodGrid.area [pix] = 0.0;
            }
         }
         pixel_convolution_1(area, NULL, WoodGrid.area, NULL, (*lsplit0<<1)+4, theta, phi, psi, NULL, npix, WoodGrid.theta,
            WoodGrid.phi, WoodGrid.psi, WoodGrid.N, NULL, NULL, 0, 0, 0, nspace_conv, 5);
         for(pix=0;pix<WoodGrid.N;pix++) WoodGrid.area[pix] *= sin(WoodGrid.theta[pix]) * dtheta * dtheta;

         /* Which multipole number corresponds to which mode in the vector */
         for(L=0;L<*lsplit0;L++)
            for(M=-L;M<=L;M++)
               MultipoleNum[L*(L+1)+M] = L;

         /* The transfer matrix is computed one column at a time */
         for(i=0;i<*Nmodes;i++) {

            /* Build a unit column vector for use in the preconditioner.  The WhiteNoiseLevel
             * in the denominator provides the "N^-1" in the Woodbury formula.
             */
            for(j=0;j<*Nmodes;j++) YLMOut.vector[j] = YLMIn.vector[j] = 0.;
            YLMIn.vector[i] = (*sqrt_Cl_prior)[L = MultipoleNum[i]] / (*WhiteNoiseLevel);

            /* Now do spherical convolution on set of pixels.  Note that we
             * are going from harmonic-->real-->harmonic space.  We will use
             * pm_order = 3.
             */
            for(j=0;j<WoodGrid.N;j++) ModeMap[j] = 0.;
            pixel_synthesis_1(&YLMIn, NULL, ModeMap, NULL, *lsplit0-1, WoodGrid.theta, WoodGrid.phi, WoodGrid.psi,
               WoodGrid.N, NULL, NULL, 0, 0, nspace_conv, 3);
            for(j=0;j<WoodGrid.N;j++) ModeMap[j] *= WoodGrid.area[j];
            pixel_analysis_1(&YLMOut, NULL, ModeMap, NULL, *lsplit0-1, WoodGrid.theta, WoodGrid.phi, WoodGrid.psi,
               NULL, WoodGrid.N, *sqrt_Cl_prior, NULL, 0, 1, nspace_conv, 3);

            /* Store convolution results in transfer matrix */
            for(j=0;j<*Nmodes;j++) (*TransferMatrix)[i][j] = YLMOut.vector[j];

         }

         /* Now we have computed V' N^-1 U.  We want to get (1 + V' N^-1 U)^-1, which
          * is obtained by adding the identity and inverting.
          */
         for(i=0;i<*Nmodes;i++) (*TransferMatrix)[i][i] += 1.;
         gaussjinv(*TransferMatrix,*Nmodes);

         /* We won't be needing these anymore so kill them */
         free_lvector(MultipoleNum,0,*Nmodes-1);
         free_dvector(ModeMap, 0, WoodGrid.N-1);
         deallocate_sphere_pixel(&WoodGrid);
         deallocate_sphere_modes(&YLMIn );
         deallocate_sphere_modes(&YLMOut);

         /* Keep only diagonal and below-diagonal entries. */
         wtMatrix[tnum].nsparse = (*Nmodes * (*Nmodes+1)) /2;
         wtMatrix[tnum].element = dvector(0, wtMatrix[tnum].nsparse-1);

         k = 0;
         for(i=0;i<*Nmodes;i++) for(j=0;j<=i;j++) {
            wtMatrix[tnum].element[k] = (*TransferMatrix)[j][i];
            k++;
         }

         /* Cleanup and exit */
         free_dmatrix(*TransferMatrix, 0, *Nmodes-1, 0, *Nmodes-1);         
         return;

      case 0: /* Actually do a C^{-1}X computation using Woodbury */

         /* Allocate memory for intermediate steps */
         allocate_sphere_modes(&YLMIn , *lsplit0-1);
         allocate_sphere_modes(&YLMOut, *lsplit0-1);
         XIntermediate = dvector(0,npix-1);

         /* Do Woodbury formula.  First step is to compute XIntermediate = N^-1 X,
          * then YLMIn = V' N^-1 X.
          */
         for(i=0;i<npix;i++) XIntermediate[i] = X[i] / (*WhiteNoiseLevel) * area[i];
         pixel_analysis_1(&YLMIn, NULL, XIntermediate, NULL, *lsplit0-1, theta, phi, psi, NULL, npix, *sqrt_Cl_prior,
            NULL, 0, 1, nspace_conv, 3);
         for(i=0;i<npix;i++) XIntermediate[i] = 0.;

         /* Multiply by transfer matrix to get YLMOut = (1 + V' N^-1 U)^-1 V' N^-1 X. */
         ep = wtMatrix[tnum].element;
         for(i=0;i<*Nmodes;i++) for(j=0;j<=i;j++) {
            if (i!=j) YLMOut.vector[j] += *ep * YLMIn.vector[i];
            YLMOut.vector[i] += *(ep++) * YLMIn.vector[j];
         }

         /* Finish by computing ApproxCInvX = (N+UV')^-1 X. */
         pixel_synthesis_1(&YLMOut, NULL, XIntermediate, NULL, *lsplit0-1, theta, phi, psi, npix, *sqrt_Cl_prior, NULL,
            0, 1, nspace_conv, 3);
         for(i=0;i<npix;i++) ApproxCInvX[i] += (X[i] - XIntermediate[i]) / (*WhiteNoiseLevel) * area[i];

         /* Clean up memory */
         deallocate_sphere_modes(&YLMIn );
         deallocate_sphere_modes(&YLMOut);
         free_dvector(XIntermediate,0,npix-1);
         return;

      default: /* We'd better not end up here */
            fprintf(stderr, "Error: unknown OpSelect code %d", OpSelect);
            exit(1);
   }
}
#endif

/* pixel_analysis_1_wrapper
 * *** SHT ANALYSIS ON A SET OF PIXELS ***
 * (WRAPPER FOR pixel_analysis_1)
 *
 * Performs harmonic => real space conversion on a set of pixels.  The result is added to the output
 * map, not over-written.
 *
 * Arguments:
 *   p_LM: + parity (S, V, or E) component of harmonic representation of the function
 *   n_LM: - parity (A or B) component of harmonic representation of the function
 *      (not used for scalar transform, can pass NULL)
 * > component_1: the real-space representation (S, X, or Q component)
 * > component_2: the real-space representation (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 *   lmax: maximum multipole used in the transform
 *   theta: colatitude vector of pixels
 *   phi: longitude vector of pixels
 *   psi: orientation vector of pixels (0 = X to East, Y to North; pi/2 = X to North, Y to West)
 *   area: area vector of pixels (only used if convolve_flag 0x2 bit is on)
 *   length: number of pixels used for transform
 *   convolve_kernel_p: convolution kernel used for + parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   convolve_kernel_n: convolution kernel used for - parity components [0..lmax].  Used if convolve_flag
 *      0x1 bit is on.
 *   spin: type of transform: 0 = scalar, 1 = vector, 2 = tensor.
 *   convolve_flag: two bits: 0x1 = apply convolution kernel? [Yes=1, No=0]
 *                            0x2 = multiply by area? [Yes=1, No=0]
 *   nspace: size of grid used for interpolation
 *   pm_order: order of PM interpolation
 */

void pixel_analysis_1_wrapper(double *p_LM, double *n_LM, double *component_1, double *component_2,
   long lmax, double *theta, double *phi, double *psi, double *area, long length, double *convolve_kernel_p,
   double *convolve_kernel_n, int spin, unsigned short int convolve_flag, long nspace, long pm_order) {

   SPHERE_MODES p_LM_struct, n_LM_struct;
   long L;

   /* Generate SPHERE_MODES structures -- positive parity first */
   p_LM_struct.lmax = lmax;
   p_LM_struct.Nmode = (lmax+1)*(lmax+1);
   p_LM_struct.vector = p_LM;
   p_LM_struct.coefs = (double**)malloc((size_t)((lmax+1)*sizeof(double*)));
   for(L=0;L<=lmax;L++)
      p_LM_struct.coefs[L] = p_LM_struct.vector + L*(L+1);

   /* and now negative parity */
   if (spin) {
      n_LM_struct.lmax = lmax;
      n_LM_struct.Nmode = (lmax+1)*(lmax+1);
      n_LM_struct.vector = n_LM;
      n_LM_struct.coefs = (double**)malloc((size_t)((lmax+1)*sizeof(double*)));
      for(L=0;L<=lmax;L++)
         n_LM_struct.coefs[L] = n_LM_struct.vector + L*(L+1);
   }

   /* Run pixel_analysis_1 */
   pixel_analysis_1(&p_LM_struct, &n_LM_struct, component_1, component_2, lmax, theta, phi,
      psi, area, length, convolve_kernel_p, convolve_kernel_n, spin, convolve_flag, nspace, pm_order);

   /* Cleanup.  Note that we are freeing the coefs array of pointers but we don't free the
    * data itself!
    */
   free( (char *) p_LM_struct.coefs );
   if (spin) free( (char *) n_LM_struct.coefs );
}

/* pixel_synthesis_1_wrapper
 * *** SHT SYNTHESIS ON A SET OF PIXELS ***
 * (WRAPPER FOR pixel_synthesis_1)
 *
 * Performs harmonic => real space conversion on a set of pixels.  The result is added to the output map,
 * not over-written.
 *
 * Arguments:
 *   p_LM: + parity (S, V, or E) component of harmonic representation of the function
 *   n_LM: - parity (A or B) component of harmonic representation of the function
 *      (not used for scalar transform, can pass NULL)
 * > component_1: the real-space representation (S, X, or Q component)
 * > component_2: the real-space representation (Y or U component -- not used for scalar transform,
 *      can pass NULL)
 *   lmax: maximum multipole used in the transform
 *   theta: colatitude vector of pixels
 *   phi: longitude vector of pixels
 *   psi: orientation vector of pixels (0 = X to East, Y to North; pi/2 = X to North, Y to West)
 *   length: number of pixels used for transform
 *   convolve_kernel_p: convolution kernel used for + parity components [0..lmax].  Used if convolve_flag==1,
 *      and not if convolve_flag==0.
 *   convolve_kernel_n: convolution kernel used for - parity components [0..lmax].  Used if convolve_flag==1,
 *      and not if convolve_flag==0.
 *   spin: type of transform: 0 = scalar, 1 = vector, 2 = tensor.
 *   convolve_flag: apply convolution kernel?  [Yes=1, No=0]
 *   nspace: size of grid used for interpolation
 *   pm_order: order of PM interpolation
 */

void pixel_synthesis_1_wrapper(double *p_LM, double *n_LM, double *component_1, double *component_2,
   long lmax, double *theta, double *phi, double *psi, long length, double *convolve_kernel_p,
   double *convolve_kernel_n, int spin, unsigned short int convolve_flag, long nspace, long pm_order) {

   SPHERE_MODES p_LM_struct, n_LM_struct;
   long L;

   /* Generate SPHERE_MODES structures -- positive parity first */
   p_LM_struct.lmax = lmax;
   p_LM_struct.Nmode = (lmax+1)*(lmax+1);
   p_LM_struct.vector = p_LM;
   p_LM_struct.coefs = (double**)malloc((size_t)((lmax+1)*sizeof(double*)));
   for(L=0;L<=lmax;L++)
      p_LM_struct.coefs[L] = p_LM_struct.vector + L*(L+1);

   /* and now negative parity */
   if (spin) {
      n_LM_struct.lmax = lmax;
      n_LM_struct.Nmode = (lmax+1)*(lmax+1);
      n_LM_struct.vector = n_LM;
      n_LM_struct.coefs = (double**)malloc((size_t)((lmax+1)*sizeof(double*)));
      for(L=0;L<=lmax;L++)
         n_LM_struct.coefs[L] = n_LM_struct.vector + L*(L+1);
   }

   /* Run pixel_synthesis_1 */
   pixel_synthesis_1(&p_LM_struct, &n_LM_struct, component_1, component_2, lmax, theta, phi,
      psi, length, convolve_kernel_p, convolve_kernel_n, spin, convolve_flag, nspace, pm_order);

   /* Cleanup.  Note that we are freeing the coefs array of pointers but we don't free the
    * data itself!
    */
   free( (char *) p_LM_struct.coefs );
   if (spin) free( (char *) n_LM_struct.coefs );
}

/* pixels_rect_1
 * *** CONVERTS PIXELS FROM (theta,phi) to (x,y,z) ***
 *
 * This function takes in the pixel spherical coordinates and returns the
 * Cartesian coordinates (x,y,z) of each pixel.  This is useful if we will be repeatedly
 * accessing the Cartesian components and don't want to call trig functions every time.
 *
 * Arguments:
 *   N: number of pixels
 *   theta: N-vector of colatitudes (radians)
 *   phi: N-vector of longitudes (radians)
 * > x: N-vector of x coordinates
 * > y: N-vector of y coordinates
 * > z: N-vector of z coordinates
 */

void pixels_rect_1(long N, double *theta, double *phi, double *x, double *y, double *z) {

   double sintheta;

   while(N>0) {
      /* spherical -> Cartesian conversion */
      sintheta = sin(*theta);
      *x = sintheta * cos(*phi);
      *y = sintheta * sin(*phi);
      *z = cos(*theta);

      /* increment pointers */
      theta++; phi++; x++; y++; z++;

      /* decrement number of points remaining */
      N--;
   }
}

/* remap_icosa_1
 * *** RE-MAPS THE FACE OF AN ICOSAHEDRON TO REDUCE MAPPING DISTORTIONS ***
 *
 * This function takes in the x and y coordinates of a point on an icosahedral
 * face and re-maps them onto a new grid that has lower distortions.  The side
 * length is Nside, and the new positions are re-written to (x,y).  The mapping
 * from the flat faces to the lower-distortion faces have mdir=1, the other way is
 * mdir=-1.
 *
 * The coordinate system on the icosahedral face is that the vertices are at
 * (0,0), (0,Nside), and (Nside,0).
 *
 * Arguments:
 * > x: coordinate on face
 * > y: coordinate on face
 *   Nside: size of face
 *   mdir: which direction to transform
 */

void remap_icosa_1(float *x, float *y, float Nside, int mdir) {

   int sextant;
   float sa, sb, sc, wc, we, wv, tot;

#ifdef N_CHECKVAL
   /* Check whether the direction is correct */
   if (mdir!= 1 && mdir!= -1) {
      fprintf(stderr, "Error in remap_icosa_1: mdir=%d is illegal value.\n", mdir);
      exit(1);
   }
#endif

   /* We first need to figure out which sextant we are looking at.   The sextants
    * are: 1 = (0,0) ... (Nside/2,0) ... (Nside/3,Nside/3)
    *      2 = (Nside/2,0) ... (Nside,0) ... (Nside/3,Nside/3)
    *      3 = (Nside/2,Nside/2) ... (Nside,0) ... (Nside/3,Nside/3)
    *      4 = (Nside/2,Nside/2) ... (0,Nside) ... (Nside/3,Nside/3)
    *      5 = (0,Nside/2) ... (0,Nside) ... (Nside/3,Nside/3)
    *      6 = (0,Nside/2) ... (0,0) ... (Nside/3,Nside/3)
    */
   sa = 1 - (2* *x + *y)/Nside;
   sb = (*x - *y)/Nside;
   sc = -1 + (*x + 2* *y)/Nside;
   sextant = sa>0? (sb>0? 1: sc>0? 5: 6): (sb<0? 4: sc>0? 3: 2);

   /* Coordinates in this sextant */
   switch(sextant) {
      case 1:
         we = 2*sb;
         wv = sa;
         break;
      case 2:
         we = -2*sc;
         wv = -sa;
         break;
      case 3:
         we = 2*sc;
         wv = sb;
         break;
      case 4:
         we = -2*sa;
         wv = -sb;
         break;
      case 5:
         we = 2*sa;
         wv = sc;
         break;
      case 6:
         we = -2*sb;
         wv = -sc;
         break;
      default:
         /* We shouldn't get here */
         fprintf(stderr, "Error in remap_icosa_1: you can't get here: switch(sextant) --> default [1].\n");
         exit(1);
         break;
   } /* End switch(sextant) */
   wc = 1 - we - wv;

   /* Switch to primed system.  Here RHO = center-to-vertex angle of icosahedron, and
    * VARTHETA = vertex-to-middle-of-edge angle of icosahedron.
    */
#define COSRHO 0.7946544722918
#define COSVARTHETA 0.850650808352
   if (mdir>0) {
      we *= COSVARTHETA;
      wc *= COSRHO;
   } else {
      we /= COSVARTHETA;
      wc /= COSRHO;
   }
#undef COSRHO
#undef COSVARTHETA
   tot = wc+we+wv;

   /* Compute weights.  Note that these are actually Nside*wc/3, Nside*we/2, and Nside*wv,
    * since we use these combinations in what follows.
    */
   wc *= Nside/(3.*tot);
   we *= Nside/(2.*tot);
   wv *= Nside/tot;

   /* Now let's go from the weights back to (x,y) */
   switch(sextant) {
      case 1:
         *y = wc;
         *x = we + wc;
         break;
      case 2:
         *x = wv + we + wc;
         *y = wc;
         break;
      case 3:
         *x = wv + we + wc;
         *y = wc + we;
         break;
      case 4:
         *x = we + wc;
         *y = wv + we + wc;
         break;
      case 5:
         *x = wc;
         *y = wv + we + wc;
         break;
      case 6:
         *x = wc;
         *y = we + wc;
         break;
      default:
         /* We shouldn't get here */
         fprintf(stderr, "Error in remap_icosa_1: you can't get here: switch(sextant) --> default [2].\n");
         exit(1);
         break;
   } /* End switch(sextant) */
}

/* pixels_icosa_1
 * *** COMPUTES ICOSAHEDRAL INDICES OF EACH PIXEL GIVEN (theta,phi) ***
 *
 * This function takes the pixel spherical coordinates (theta,phi) and converts them
 * into icosahedral pixel indices (0 .. Npix-1) where Npix=10*Nside*(Nside+1) is the
 * number of pixels in the icosahedron.
 *
 * Arguments:
 *   N: number of pixels
 *   theta: N-vector of colatitudes (radians)
 *   phi: N-vector of longitudes (radians)
 *   Nside: side length of icosahedral face (in pixels)
 * > icosapix: N-vector of icosahedral pixel indices (in .vector[icosapix[i]] format,
 *      not .data[f][x][y])
 */

void pixels_icosa_1(long N, double *theta, double *phi, long Nside, long *icosapix) {

   double sintheta, x, y, z, x36, y36, z36, x60, y60, z60, x72, y72, z72;
   double x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4;
   int face;
   float alt_ab, alt_bc, alt_ca, Nside__alt_tot, xfloat, yfloat, xfrac, yfrac;
   long xint, yint;
   long Npix_per_face = ((Nside+1)*(Nside+2)) >>1;
   long TwoNsidePlusThree = 2*Nside+3;

#define EPS_ICOSA 0.0000001

   while(N>0) {
      /* spherical -> Cartesian conversion */
      sintheta = sin(*theta);
      x = sintheta * cos(*phi);
      y = sintheta * sin(*phi);
      z = cos(*theta);

      /* Compute useful products -- to save time we only want to do these once */
      x36 = Cos36Deg * x;  y36 = Cos36Deg * y;  z36 = Cos36Deg * z;
      x60 = 0.5      * x;  y60 = 0.5      * y;  z60 = 0.5      * z;
      x72 = Cos72Deg * x;  y72 = Cos72Deg * y;  z72 = Cos72Deg * z;

      /* Now that we have Cartesian components, go after which face we're on.
       * Begin by rotating 72 degrees around A=( sqrt(1/2+sqrt(5)/10), sqrt(1/2-sqrt(5)/10), 0).
       */
      x1 =  x36 +y72 -z60;
      y1 =  x72 +y60 +z36;
      z1 =  x60 -y36 +z72;

      /* Rotation by 144 degrees */
      x2 =  x60 +y36 -z72;
      y2 =  x36 -y72 +z60;
      z2 =  x72 -y60 -z36;

      /* Rotation by 216 degrees */
      x3 =  x60 +y36 +z72;
      y3 =  x36 -y72 -z60;
      z3 = -x72 +y60 -z36;

      /* Rotation by 288 degrees */
      x4 =  x36 +y72 +z60;
      y4 =  x72 +y60 -z36;
      z4 = -x60 +y36 +z72;

      /* Now test for which face we are on.  Begin with the first band of faces around A.  The
       * purpose of the EPS_ICOSA things is so that pixels that lie near boundaries don't accidentally
       * get assigned to no face at all.
       */
      if        (z <= EPS_ICOSA && x3>=-EPS_ICOSA && z1>=-EPS_ICOSA) {face =  1; alt_ab = -z ; alt_bc =  x3; alt_ca =  z1;
      } else if (z1<= EPS_ICOSA && x4>=-EPS_ICOSA && z2>=-EPS_ICOSA) {face =  2; alt_ab = -z1; alt_bc =  x4; alt_ca =  z2;
      } else if (z2<= EPS_ICOSA && x >=-EPS_ICOSA && z3>=-EPS_ICOSA) {face =  3; alt_ab = -z2; alt_bc =  x ; alt_ca =  z3;
      } else if (z3<= EPS_ICOSA && x1>=-EPS_ICOSA && z4>=-EPS_ICOSA) {face =  4; alt_ab = -z3; alt_bc =  x1; alt_ca =  z4;
      } else if (z4<= EPS_ICOSA && x2>=-EPS_ICOSA && z >=-EPS_ICOSA) {face =  5; alt_ab = -z4; alt_bc =  x2; alt_ca =  z ;
      } else if (x3<= EPS_ICOSA && y4>=-EPS_ICOSA && y2>=-EPS_ICOSA) {face =  6; alt_ab = -x3; alt_bc =  y4; alt_ca =  y2;
      } else if (x4<= EPS_ICOSA && y >=-EPS_ICOSA && y3>=-EPS_ICOSA) {face =  7; alt_ab = -x4; alt_bc =  y ; alt_ca =  y3;
      } else if (x <= EPS_ICOSA && y1>=-EPS_ICOSA && y4>=-EPS_ICOSA) {face =  8; alt_ab = -x ; alt_bc =  y1; alt_ca =  y4;
      } else if (x1<= EPS_ICOSA && y2>=-EPS_ICOSA && y >=-EPS_ICOSA) {face =  9; alt_ab = -x1; alt_bc =  y2; alt_ca =  y ;
      } else if (x2<= EPS_ICOSA && y3>=-EPS_ICOSA && y1>=-EPS_ICOSA) {face = 10; alt_ab = -x2; alt_bc =  y3; alt_ca =  y1;
      } else if (y1<= EPS_ICOSA && x >=-EPS_ICOSA && y4<= EPS_ICOSA) {face = 11; alt_ab = -y1; alt_bc =  x ; alt_ca = -y4;
      } else if (y2<= EPS_ICOSA && x1>=-EPS_ICOSA && y <= EPS_ICOSA) {face = 12; alt_ab = -y2; alt_bc =  x1; alt_ca = -y ;
      } else if (y3<= EPS_ICOSA && x2>=-EPS_ICOSA && y1<= EPS_ICOSA) {face = 13; alt_ab = -y3; alt_bc =  x2; alt_ca = -y1;
      } else if (y4<= EPS_ICOSA && x3>=-EPS_ICOSA && y2<= EPS_ICOSA) {face = 14; alt_ab = -y4; alt_bc =  x3; alt_ca = -y2;
      } else if (y <= EPS_ICOSA && x4>=-EPS_ICOSA && y3<= EPS_ICOSA) {face = 15; alt_ab = -y ; alt_bc =  x4; alt_ca = -y3;
      } else if (z >=-EPS_ICOSA && z1<= EPS_ICOSA && x3<= EPS_ICOSA) {face = 16; alt_ab =  z ; alt_bc = -z1; alt_ca = -x3;
      } else if (z1>=-EPS_ICOSA && z2<= EPS_ICOSA && x4<= EPS_ICOSA) {face = 17; alt_ab =  z1; alt_bc = -z2; alt_ca = -x4;
      } else if (z2>=-EPS_ICOSA && z3<= EPS_ICOSA && x <= EPS_ICOSA) {face = 18; alt_ab =  z2; alt_bc = -z3; alt_ca = -x ;
      } else if (z3>=-EPS_ICOSA && z4<= EPS_ICOSA && x1<= EPS_ICOSA) {face = 19; alt_ab =  z3; alt_bc = -z4; alt_ca = -x1;
      } else if (z4>=-EPS_ICOSA && z <= EPS_ICOSA && x2<= EPS_ICOSA) {face = 20; alt_ab =  z4; alt_bc = -z ; alt_ca = -x2;
      } else {
         /* We shouldn't be able to get here. */
         fprintf(stderr, "Error in pixels_icosa_1: can't get here; increase EPS_ICOSA.\n");
         fprintf(stderr, "Location:   (%lf, %lf, %lf)\n", x, y, z);
         fprintf(stderr, "after rot1: (%lf, %lf, %lf)\n", x1, y1, z1);
         fprintf(stderr, "after rot2: (%lf, %lf, %lf)\n", x2, y2, z2);
         fprintf(stderr, "after rot3: (%lf, %lf, %lf)\n", x3, y3, z3);
         fprintf(stderr, "after rot4: (%lf, %lf, %lf)\n", x4, y4, z4);
         exit(1);
      }

      /* If the altitude is slightly out of range, move into triangle */
      alt_ab = alt_ab<0? 0: alt_ab;
      alt_bc = alt_bc<0? 0: alt_bc;
      alt_ca = alt_ca<0? 0: alt_ca;

      /* Now construct the total altitudes and get xfloat and yfloat */
      Nside__alt_tot = (double)Nside / (alt_ab+alt_bc+alt_ca);
      xfloat = alt_ab * Nside__alt_tot;
      yfloat = alt_bc * Nside__alt_tot;

      /* Map onto lower-distortion grid */
      remap_icosa_1(&xfloat, &yfloat, (float)Nside, 1);

      /* Break into integer and fractional parts */
      xfrac = xfloat - (xint=(long)floor(xfloat));
      yfrac = yfloat - (yint=(long)floor(yfloat));

      /* Decide which pixel to assign to. */
      switch ((int)floor(3*xfrac)) {
         case 0:
            if (yfrac>0.5-0.5*xfrac) yint++;
            break;
         case 1:
            if (yfrac>=xfrac) yint++;
            break;
         case 2:
            if (yfrac>1.0-0.5*xfrac) yint++;
            break;
         default:
            /* We shouldn't get here since 0<=xfrac<1 */
#ifdef N_CHECKVAL
            fprintf(stderr, "Error in pixels_icosa_1: you shouldn't be here: switch((int)floor(3*xfrac))-->default.\n");
            exit(1);
#endif
            break;
      }
      switch ((int)floor(3*yfrac)) {
         case 0:
            if (xfrac>0.5-0.5*yfrac) xint++;
            break;
         case 1:
            if (xfrac>yfrac) xint++;
            break;
         case 2:
            if (xfrac>1.0-0.5*yfrac) xint++;
            break;
         default:
            /* We shouldn't get here since 0<=yfrac<1 */
#ifdef N_CHECKVAL
            fprintf(stderr, "Error in pixels_icosa_1: you shouldn't be here: switch((int)floor(3*yfrac))-->default.\n");
            exit(1);
#endif
            break;
      }

      /* Now we've found the face, xint, and yint values, so we just need to find the vector index */
#ifdef N_CHECKVAL
      if (xint<0 || yint<0 || xint+yint>Nside) {
         fprintf(stderr, "Error: pixel out of range (%ld,%ld)\n", xint, yint);
      }
#endif
      *icosapix = (face-1)*Npix_per_face + ((TwoNsidePlusThree-xint)*xint >> 1) + yint;

      theta++; phi++; icosapix++; /* increment pointers */
      N--;                        /* decrement number of points remaining */
   }

#undef EPS_ICOSA
}

/* make_icosa_grid_1
 * *** GENERATES AN ICOSAHEDRAL GRID ***
 *
 * Constructs tables of (theta,phi) for the icosahedral grid.  The arrays theta and
 * phi are allocated by the routine.  Note that we have to pass pointers to theta and
 * phi so that the theta and phi pointers of the calling routine will point to the
 * allocated memory.
 *
 * Arguments:
 *   Nside: side length
 * > icosa: SPHERE_PIXEL structure containing the icosahedral pixelization.
 *      Warning: doesn't allocate icosa->psi or icosa->area, the user has to do this
 *      him/herself if this info is wanted.
 */
#ifndef THREAD_SAFE_SHT
void make_icosa_grid_1(SPHERE_PIXEL *icosa, long Nside) {

   /* Coordinates of the normals to the edges of the icosahedron.  Each array is for faces 1..20;
    * the Cartesian components x, y, and z are considered separately.  The edges of each face are
    * ab, bc, and ca, and the normals are taken to be on the side occupied by the face.
    */
   static double eabx[] = {0., -0.5, -Cos72Deg, Cos72Deg, 0.5, -0.5, -Cos36Deg, -1., -Cos36Deg, -0.5,
      -Cos72Deg, -Cos36Deg, -Cos36Deg, -Cos72Deg, 0., 0., 0.5, Cos72Deg, -Cos72Deg, -0.5};
   static double eaby[] = {0., Cos36Deg, 0.5, -0.5, -Cos36Deg, -Cos36Deg, -Cos72Deg, 0., -Cos72Deg, -Cos36Deg,
      -0.5, Cos72Deg, Cos72Deg, -0.5, -1., 0., -Cos36Deg, -0.5, 0.5, Cos36Deg};
   static double eabz[] = {-1., -Cos72Deg, Cos36Deg, Cos36Deg, -Cos72Deg, -Cos72Deg, -0.5, 0., 0.5, Cos72Deg,
      -Cos36Deg, -0.5, 0.5, Cos36Deg, 0., 1., Cos72Deg, -Cos36Deg, -Cos36Deg, Cos72Deg};
   static double ebcx[] = {0.5, Cos36Deg, 1., Cos36Deg, 0.5, Cos72Deg, 0., Cos72Deg, Cos36Deg, Cos36Deg,
      1., Cos36Deg, 0.5, 0.5, Cos36Deg, -0.5, -Cos72Deg, Cos72Deg, 0.5, 0.};
   static double ebcy[] = {Cos36Deg, Cos72Deg, 0., Cos72Deg, Cos36Deg, 0.5, 1., 0.5, -Cos72Deg, -Cos72Deg,
      0., Cos72Deg, Cos36Deg, Cos36Deg, Cos72Deg, Cos36Deg, 0.5, -0.5, -Cos36Deg, 0.};
   static double ebcz[] = {Cos72Deg, 0.5, 0., -0.5, -Cos72Deg, -Cos36Deg, 0., Cos36Deg, 0.5, -0.5,
      0., -0.5, -Cos72Deg, Cos72Deg, 0.5, -Cos72Deg, Cos36Deg, Cos36Deg, -Cos72Deg, -1.};
   static double ecax[] = {0.5, Cos72Deg, -Cos72Deg, -0.5, 0., Cos36Deg, Cos36Deg, Cos72Deg, 0., Cos72Deg,
      -Cos72Deg, 0., -Cos72Deg, -Cos36Deg, -Cos36Deg, -0.5, -Cos36Deg, -1., -Cos36Deg, -0.5};
   static double ecay[] = {-Cos36Deg, -0.5, 0.5, Cos36Deg, 0., -Cos72Deg, -Cos72Deg, 0.5, 1., 0.5,
      -0.5, -1., -0.5, Cos72Deg, Cos72Deg, -Cos36Deg, -Cos72Deg, 0., -Cos72Deg, -Cos36Deg};
   static double ecaz[] = {Cos72Deg, -Cos36Deg, -Cos36Deg, Cos72Deg, 1., 0.5, -0.5, -Cos36Deg, 0., Cos36Deg,
      Cos36Deg, 0., -Cos36Deg, -0.5, 0.5, -Cos72Deg, -0.5, 0., 0.5, Cos72Deg};

   float xfloat, yfloat;
   long xint, yint;
   int face__1;
   double a,b,c,offset,x,y,z;
   double *thetaptr, *phiptr, *areaptr;
   double area0;

   icosa->area_flag = 1;
   icosa->N = 10 * (Nside+1) * (Nside+2);
   area0 = FourPi / (10.0 * Nside * Nside);

   /* Allocate memory */
   thetaptr = icosa->theta = dvector(0, icosa->N-1);
   phiptr   = icosa->phi   = dvector(0, icosa->N-1);
   areaptr  = icosa->area  = dvector(0, icosa->N-1);

   /* We have to do something trivial for psi or the de-allocation routine will complain. */
   icosa->psi = dvector(0,0);

   offset = Cos36Deg * Nside; /* offset in computing a,b,c below */

   /* Loop over pixels */
   for(face__1=0;face__1<20;face__1++) {
      for(xint=0;xint<=Nside;xint++) {
         for(yint=0;xint+yint<=Nside;yint++) {

            /* Map onto flat icosahedral face */
            xfloat = (float)xint;
            yfloat = (float)yint;
            remap_icosa_1(&xfloat, &yfloat, Nside, -1);

            /* For each pixel, figure out the coordinates */
            a = offset + yfloat;
            b = offset + Nside - xfloat - yfloat;
            c = offset + xfloat;

            /* Now get the Cartesian and then the polar coordinates */
            x = a * ebcx[face__1] + b * ecax[face__1] + c * eabx[face__1];
            y = a * ebcy[face__1] + b * ecay[face__1] + c * eaby[face__1];
            z = a * ebcz[face__1] + b * ecaz[face__1] + c * eabz[face__1];
            *thetaptr = atan2(sqrt(x*x+y*y), z);
            *phiptr   = atan2(y, x);

            /* Find effective area */
            *areaptr = area0;
            if (xint==0) *areaptr /= 2.0;
            if (yint==0) *areaptr /= 2.0;
            if (xint+yint==Nside) *areaptr /= 2.0;
            if (xint==0 && yint==0) *areaptr /= 1.5;
            if (xint==0 && yint==Nside) *areaptr /= 1.5;
            if (xint==Nside && yint==0) *areaptr /= 1.5;

            /* Set phi in appropriate range, increment pointers */
            if (*phiptr<0) *phiptr += TwoPi;
            thetaptr++; phiptr++; areaptr++;
         }      /* End yint loop */
      }         /* End xint loop */
   }            /* End face loop */
}
#endif

/* fourier_2d_1
 * *** 2D FFT ON A MATRIX ***
 *
 * Takes the matrices Areal[0..nx-1][0..ny-1] and Aimag[0..nx-1][0..ny-1]
 * and computes the Fourier transform on both axes.  These matrices are over-written
 * with the results.
 *
 * Arguments:
 *   nx: number of pixels in x-direction
 *   ny: number of pixels in y-direction
 * > Areal: real part of data
 * > Aimag: imag part of data
 *   isign: which direction (+1 or -1)
 */

void fourier_2d_1(long nx, long ny, double **Areal, double **Aimag, int isign) {

   long x,y;
   double *FourierStack, *fsptr, *arptr, *aiptr;
   long length_max = nx>ny? nx: ny;

   FourierStack = dvector(0, 2*length_max-1);

   /* FFT the y-direction */
   for(x=0;x<nx;x++) {
      fsptr = FourierStack;
      arptr = Areal[x];
      aiptr = Aimag[x];
      for(y=0;y<ny;y++) {
         *fsptr = *(arptr++);
         fsptr++;
         *fsptr = *(aiptr++);
         fsptr++;
      }
      fourier_trans_2(FourierStack-1, ny, isign);
      fsptr = FourierStack;
      arptr = Areal[x];
      aiptr = Aimag[x];
      for(y=0;y<ny;y++) {
         *arptr = *(fsptr++);
         arptr++;
         *aiptr = *(fsptr++);
         aiptr++;
      }
   } /* End for(x) */

   /* FFT the x-direction */
   for(y=0;y<ny;y++) {
      fsptr = FourierStack;
      for(x=0;x<nx;x++) {
         *fsptr = Areal[x][y];
         fsptr++;
         *fsptr = Aimag[x][y];
         fsptr++;
      }
      fourier_trans_2(FourierStack-1, nx, isign);
      fsptr = FourierStack;
      for(x=0;x<nx;x++) {
         Areal[x][y] = *(fsptr++);
         Aimag[x][y] = *(fsptr++);
      }
   } /* End for(x) */

   /* Clean up memory */
   free_dvector(FourierStack, 0, 2*length_max-1);
}

/* kernel_icosa_1
 * *** COMPUTES ICOSAHEDRAL CONVOLUTION KERNEL ***
 *
 * Arguments:
 *   Nside: resolution parameter
 *   fftlen: length of the FFT (1dim; must be >2.5*Nside)
 *   lmin: minimul value of L to use in convolution
 *   lmax: maximum value of L to use in convolution
 *   Cl: power spectrum for use in convolution
 * > kernel_real: Fourier-space convolution kernel (real part)
 * > kernel_imag: Fourier-space convolution kernel (imag part)
 */

void kernel_icosa_1(long Nside, long fftlen, long lmin, long lmax, double *Cl, double **kernel_real,
   double **kernel_imag) {

#define LENSIDE 1.20459100585525470727876
#define THETA_MAX_SEP 0.6022955029276

   long x,y,L,itheta;
   double *Ctheta, *Clreduced;
   double Pltheta, Plm1theta, Plp1theta, costheta, phase, itheta_pix, itheta_frac, value;
   double dtheta = PiOverTwo / lmax;
   long ntheta = (long)ceil(THETA_MAX_SEP/dtheta) + 2;
   double pixsep = LENSIDE/Nside;
   double pixwinfactor = Pi / (15.0 * Nside * Nside);

#ifdef N_CHECKVAL
   /* We need to test for fftlen being too small because this could cause wrap-around
    * effects in the FFT when fftconv_icosa_1 is called.
    */
   if (fftlen<<1<5*Nside) {
      fprintf(stderr, "Error in kernel_icosa_1: fftlen=%ld, Nside=%ld are illegal values.\n", fftlen, Nside);
      exit(1);
   }

   /* Also check that lmin is bigger than ICOSA_DL since we have to smoothly cut off
    * the kernel at lmin.
    */
   if (lmin<=ICOSA_DL) {
      fprintf(stderr, "Error in kernel_icosa_1: lmin=%ld must exceed ICOSA_DL=%d\n", lmin, ICOSA_DL);
      exit(1);
   }
#endif

   /* Allocate real-space covariance and clear kernel */
   Ctheta = dvector(-1,ntheta);
   for(x=0;x<fftlen;x++)
      for(y=0;y<fftlen;y++)
         kernel_real[x][y] = kernel_imag[x][y] = 0.;

   /* Compute reduced power spectrum (i.e. excluding portions being computed by
    * direct SHT, and including the pixel window function).
    */
   Clreduced = dvector(0, lmax);
   for(L=lmin-ICOSA_DL; L<=lmax; L++) Clreduced[L] = Cl[L] * exp (-pixwinfactor*L*L);
   for(L=lmin-ICOSA_DL; L<lmin; L++) Clreduced[L] *= 0.5 + 0.5 * cos(Pi*(lmin-L) / (double)ICOSA_DL);

   /* Populate real-space covariance by computing:
    * sum_l (2l+1)C_l/4pi * P_l(cos theta).
    */
   for(itheta=0;itheta<ntheta;itheta++) {
      costheta = cos(itheta*dtheta);
      Pltheta = 1;
      Plm1theta = 0;
      for(L=0;L<=lmax;L++) {
         if (L>=lmin) Ctheta[itheta] += Pltheta * (2*L+1) / FourPi * Clreduced[L];
         Plp1theta = ((2*L+1)*Pltheta*costheta - L*Plm1theta) / (L+1);
         Plm1theta = Pltheta;
         Pltheta = Plp1theta;
      }

      /* Impose smooth cutoff at large radii */
      if (itheta>ntheta*2.0/3.0) {
         phase = 1.5*Pi*(1.0 - itheta/(double)ntheta);
         Ctheta[itheta] *= 0.5*(1-cos(phase));
      }
   } /* End for(itheta) */

   /* Now put the real-space covariance onto the kernel grid.  First do Sextant I: */
   for(x=0; x<Nside>>1; x++) {
      for(y=0; y<=x && x+y<Nside>>1; y++) {
         itheta_pix = pixsep/dtheta * sqrt((x+y)*(x+y)-x*y);
         itheta_frac = itheta_pix - (itheta = (long)floor(itheta_pix));
#ifdef N_CHECKVAL
         if (itheta>=ntheta || itheta<-1) {
            fprintf(stderr, "Error in kernel_icosa_1: itheta=%ld out of range [-1,%ld]\n", itheta, ntheta-1);
            exit(1);
         }
#endif
         kernel_real[x][y] = kernel_real[y][x] = Ctheta[itheta] + itheta_frac*(Ctheta[itheta+1]-Ctheta[itheta]);
      } /* End for(y) */
   } /* End for(x) */

   /* Loop over the other sextants */
   for(x=0; x<Nside>>1; x++) {
      for(y=0; x+y<Nside>>1; y++) {
         value = kernel_real[x][y]; /* so we don't have to keep looking this up */

         if (y)
            kernel_real[fftlen-y][x+y] =         /* Sextant II  */
            kernel_real[fftlen-x-y][x] = value;  /* Sextant III */
         if (x && y)
            kernel_real[fftlen-x][fftlen-y] =    /* Sextant IV  */
            kernel_real[x+y][fftlen-x] = value;  /* Sextant VI  */
         if (x || y)
            kernel_real[y][fftlen-x-y] = value;  /* Sextant V   */
      } /* End for(y) */
   } /* End for(x) */

   free_dvector(Ctheta, -1, ntheta);   /* Clean up */

   /* 2D fourier transform */
   fourier_2d_1(fftlen, fftlen, kernel_real, kernel_imag, 1);

#undef LENSIDE
#undef THETA_MAX_SEP
}

/* fftconv_icosa_1
 * *** FFT CONVOLUTION ON AN ICOSAHEDRON ***
 *
 * Takes the given Fourier-space kernel (produced by conv_kernel_icosa_1) and
 * performs a flat-sky convolution on the icosahedral data indata.  The result
 * is added to outdata.  Nside must be even.
 *
 * Arguments:
 *   indata: input data
 * > outdata: output data
 *   fftlen: size of FFT plane (must be >2.5*Nside)
 *   kernel_real: real part of kernel
 *   kernel_imag: imag part of kernel
 */

void fftconv_icosa_1(ICOSAHEDRAL *indata, ICOSAHEDRAL *outdata, long fftlen, double **kernel_real,
   double **kernel_imag) {

   int face, face0, face1, face2, face3, fcounter, adjindex;
   double **face_ptr, **face_ptrS1, **face_ptrS2;
   double tempr, tempi;
   long x,y,ymax;
   long Nside = indata->Nside;
   long HalfNside = Nside >> 1;
   long TwoNside = Nside << 1;
   long OneAndHalfNside = HalfNside+Nside;
   double norm;
   double fracm, fracp, temp;
   double **fourierplane_real, **fourierplane_imag;
   double *fptr_real, *fptr_imag, *fptr_real_inv, *fptr_imag_inv, *kptr_real, *kptr_imag;
   double *map_ptr_0, *map_ptr_1, *map_ptr_2, *map_ptr_3;
   double **map_0, **map_1, **map_2, **map_3;
   int *adjptr;
   char *rotadjptr;
   ICOSAHEDRAL dataS1, dataS2; /* rotated data */

   /* Table of adjacent faces */
   int adj1[] = { 5,  1,  2,  3,  4,  1,  2,  3,  4,  5, 10,  6,  7,  8,  9, 20, 16, 17, 18, 19};
   int adj2[] = {10,  6,  7,  8,  9,  5,  1,  2,  3,  4, 15, 11, 12, 13, 14, 19, 20, 16, 17, 18};
   int adj3[] = {11, 12, 13, 14, 15, 10,  6,  7,  8,  9, 17, 18, 19, 20, 16, 18, 19, 20, 16, 17};
   int adj4[] = { 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 16, 17, 17, 18, 19, 20, 16};
   int adj5[] = {12, 13, 14, 15, 11, 18, 19, 20, 16, 17, 19, 20, 16, 17, 18, 15, 11, 12, 13, 14};
   int adj6[] = { 7,  8,  9, 10,  6, 19, 20, 16, 17, 18, 12, 13, 14, 15, 11,  9, 10,  6,  7,  8};
   int adj7[] = { 2,  3,  4,  5,  1, 12, 13, 14, 15, 11,  6,  7,  8,  9, 10, 14, 15, 11, 12, 13};
   int adj8[] = { 3,  4,  5,  1,  2,  7,  8,  9, 10,  6,  1,  2,  3,  4,  5,  8,  9, 10,  6,  7};
   int adj9[] = { 4,  5,  1,  2,  3,  2,  3,  4,  5,  1,  5,  1,  2,  3,  4, 13, 14, 15, 11, 12};

   /* Table of orientations of adjacent faces, relative to ab=horizontal */
   char rotadj1[] = "aaaaacccccaaaaaccccc";
   char rotadj2[] = "cccccbbbbbbbbbbaaaaa";
   char rotadj3[] = "bbbbbbbbbbaaaaaccccc";
   char rotadj4[] = "aaaaaccccccccccaaaaa";
   char rotadj5[] = "bbbbbaaaaabbbbbccccc";
   char rotadj6[] = "bbbbbaaaaabbbbbccccc";
   char rotadj7[] = "ccccccccccaaaaaaaaaa";
   char rotadj8[] = "aaaaabbbbbbbbbbccccc";
   char rotadj9[] = "cccccaaaaabbbbbbbbbb";

   /* Pointers to the tables */
   int *adj[] = {NULL, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, adj9};
   char *rotadj[] = {NULL, rotadj1, rotadj2, rotadj3, rotadj4, rotadj5, rotadj6, rotadj7, rotadj8, rotadj9};

   /* to avoid compiler warnings -- we assign to these before we use them */
   face0 = face1 = face2 = face3 = 0;

#ifdef N_CHECKVAL
   /* Check for consistency of input and output array sizes */
   if (indata->Nside != outdata->Nside) {
      fprintf(stderr, "Error in fftconv_icosa_1: input and output sizes (%ld vs. %ld) do not match.\n",
         indata->Nside, outdata->Nside);
      exit(1);
   }

   /* We need to test for fftlen being too small because this could cause wrap-around effects. */
   if (fftlen<<1<5*Nside) {
      fprintf(stderr, "Error in fftconv_icosa_1: fftlen=%ld, Nside=%ld are illegal values.\n", fftlen, Nside);
      exit(1);
   }

   /* Check that Nside is even */
   if (Nside & 1) {
      fprintf(stderr, "Error in fftconv_icosa_1: Nside=%ld is odd.\n", Nside);
      exit(1);
   }
#endif

   /* Build data structures rotated by 120 and 240 degrees */
   allocate_icosahedral(&dataS1, Nside);
   allocate_icosahedral(&dataS2, Nside);
   for(face=1;face<=20;face++) {
      face_ptr = indata->data[face];
      face_ptrS1 = dataS1.data[face];
      face_ptrS2 = dataS2.data[face];
      for(x=0;x<=Nside;x++)
         for(y=0;x+y<=Nside;y++)
            face_ptrS1[Nside-x-y][x] = face_ptrS2[y][Nside-x-y] = face_ptr[x][y];
   } /* End for(face,x,y) */

   fourierplane_real = dmatrix(0, fftlen-1, 0, fftlen-1);
   fourierplane_imag = dmatrix(0, fftlen-1, 0, fftlen-1);

   /* Loop over faces.  The fcounter=0..4 will indicate which quadruplet of faces is being considered.
    * The individual faces are then face = 1..20 = fcounter*4 + findex + 1;
    */
   for(fcounter=0;fcounter<5;fcounter++) {
      /* Build the Fourier plane.  Start by clearing */
      for(x=0;x<fftlen;x++) {
         fptr_real = fourierplane_real[x];
         fptr_imag = fourierplane_imag[x];
         for(y=0;y<fftlen;y++)
            *(++fptr_real) = *(++fptr_imag) = 0.;
      } /* End for(x) */

      /* Build the central triangle for each findex.  This includes setting the face
       * numbers face0 .. face3 of the triangles we're going to compute.
       */
      for(x=0;x<=Nside;x++) {
         map_ptr_0 = indata->data[face0 = (fcounter<<2)+1][x];
         map_ptr_1 = indata->data[face1 = (fcounter<<2)+2][x];
         map_ptr_2 = indata->data[face2 = (fcounter<<2)+3][x];
         map_ptr_3 = indata->data[face3 = (fcounter<<2)+4][x];
         fptr_real = fourierplane_real[x + HalfNside] + HalfNside;
         fptr_imag = fourierplane_imag[x + HalfNside] + HalfNside;
         fptr_real_inv = fourierplane_real[TwoNside-x] + TwoNside;
         fptr_imag_inv = fourierplane_imag[TwoNside-x] + TwoNside;
         ymax = Nside-x;
         for(y=0;y<=ymax;y++) {
            *fptr_real = *(map_ptr_0++);
            *fptr_imag = *(map_ptr_1++);
            *fptr_real_inv = *(map_ptr_2++);
            *fptr_imag_inv = *(map_ptr_3++);
            fptr_real++;
            fptr_imag++;
            fptr_real_inv--;
            fptr_imag_inv--;
         } /* End for(y) */
      } /* End for(x) */

      /* Now build the adjacent faces */
      for(adjindex=1; adjindex<=9; adjindex++) {

         /* Identify which adjacent face we want, and in which orientation */
         adjptr = adj[adjindex];
         rotadjptr = rotadj[adjindex];
         map_0 = rotadj[adjindex][face0-1]=='a'?  dataS1.data[adj[adjindex][face0-1]]:
                 rotadj[adjindex][face0-1]=='b'? indata->data[adj[adjindex][face0-1]]:  dataS2.data[adj[adjindex][face0-1]];
         map_1 = rotadj[adjindex][face1-1]=='a'?  dataS1.data[adj[adjindex][face1-1]]:
                 rotadj[adjindex][face1-1]=='b'? indata->data[adj[adjindex][face1-1]]:  dataS2.data[adj[adjindex][face1-1]];
         map_2 = rotadj[adjindex][face2-1]=='a'?  dataS1.data[adj[adjindex][face2-1]]:
                 rotadj[adjindex][face2-1]=='b'? indata->data[adj[adjindex][face2-1]]:  dataS2.data[adj[adjindex][face2-1]];
         map_3 = rotadj[adjindex][face3-1]=='a'?  dataS1.data[adj[adjindex][face3-1]]:
                 rotadj[adjindex][face3-1]=='b'? indata->data[adj[adjindex][face3-1]]:  dataS2.data[adj[adjindex][face3-1]];

         switch(adjindex) {
            case 1:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[x];
                  map_ptr_1 = map_1[x];
                  map_ptr_2 = map_2[x];
                  map_ptr_3 = map_3[x];
                  fptr_real = fourierplane_real[HalfNside - x] + OneAndHalfNside;
                  fptr_imag = fourierplane_imag[HalfNside - x] + OneAndHalfNside;
                  fptr_real_inv = fourierplane_real[TwoNside+x] + Nside;
                  fptr_imag_inv = fourierplane_imag[TwoNside+x] + Nside;
                  ymax = Nside-x;
                  for(y=0;y<=ymax;y++) {
                     *fptr_real += *(map_ptr_0++);
                     *fptr_imag += *(map_ptr_1++);
                     *fptr_real_inv += *(map_ptr_2++);
                     *fptr_imag_inv += *(map_ptr_3++);
                     fptr_real--;
                     fptr_imag--;
                     fptr_real_inv++;
                     fptr_imag_inv++;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 4:
               for(x=0;x<=Nside;x++) {
                  map_ptr_0 = map_0[x];
                  map_ptr_1 = map_1[x];
                  map_ptr_2 = map_2[x];
                  map_ptr_3 = map_3[x];
                  fptr_real = fourierplane_real[OneAndHalfNside - x] + HalfNside;
                  fptr_imag = fourierplane_imag[OneAndHalfNside - x] + HalfNside;
                  fptr_real_inv = fourierplane_real[Nside+x] + TwoNside;
                  fptr_imag_inv = fourierplane_imag[Nside+x] + TwoNside;
                  ymax = x>HalfNside? Nside-x: HalfNside-1;
                  for(y=0;y<=ymax;y++) {
                     *fptr_real += *(map_ptr_0++);
                     *fptr_imag += *(map_ptr_1++);
                     *fptr_real_inv += *(map_ptr_2++);
                     *fptr_imag_inv += *(map_ptr_3++);
                     fptr_real--;
                     fptr_imag--;
                     fptr_real_inv++;
                     fptr_imag_inv++;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 7:
               for(x=0;x<=Nside;x++) {
                  map_ptr_0 = map_0[x] + Nside-x; /* Must resport to some trickery since this face */
                  map_ptr_1 = map_1[x] + Nside-x; /* is upside-down.                               */
                  map_ptr_2 = map_2[x] + Nside-x;
                  map_ptr_3 = map_3[x] + Nside-x;
                  fptr_real = fourierplane_real[OneAndHalfNside-x] + HalfNside+x;
                  fptr_imag = fourierplane_imag[OneAndHalfNside-x] + HalfNside+x;
                  fptr_real_inv = fourierplane_real[Nside+x] + TwoNside-x;
                  fptr_imag_inv = fourierplane_imag[Nside+x] + TwoNside-x;
                  ymax = x>HalfNside? Nside-x: HalfNside-1;
                  for(y=0;y<=ymax;y++) {
                     *fptr_real += *(map_ptr_0--);
                     *fptr_imag += *(map_ptr_1--);
                     *fptr_real_inv += *(map_ptr_2--);
                     *fptr_imag_inv += *(map_ptr_3--);
                     fptr_real++;
                     fptr_imag++;
                     fptr_real_inv--;
                     fptr_imag_inv--;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 2:
            case 9:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[Nside-x]; /* note this is reflected */
                  map_ptr_1 = map_1[Nside-x];
                  map_ptr_2 = map_2[Nside-x];
                  map_ptr_3 = map_3[Nside-x];
                  fptr_real = fourierplane_real[HalfNside - x] + HalfNside;
                  fptr_imag = fourierplane_imag[HalfNside - x] + HalfNside;
                  fptr_real_inv = fourierplane_real[TwoNside+x] + TwoNside;
                  fptr_imag_inv = fourierplane_imag[TwoNside+x] + TwoNside;
                  if (adjindex==9) {
                     fptr_real += Nside;
                     fptr_imag += Nside;
                     fptr_real_inv -= Nside;
                     fptr_imag_inv -= Nside;
                  }
                  for(y=0;y<=x;y++) {
                     *fptr_real += *(map_ptr_0++);
                     *fptr_imag += *(map_ptr_1++);
                     *fptr_real_inv += *(map_ptr_2++);
                     *fptr_imag_inv += *(map_ptr_3++);
                     fptr_real++;
                     fptr_imag++;
                     fptr_real_inv--;
                     fptr_imag_inv--;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 3:
            case 5:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[x] + Nside-x; /* Go from "top" (highest y) down */
                  map_ptr_1 = map_1[x] + Nside-x;
                  map_ptr_2 = map_2[x] + Nside-x;
                  map_ptr_3 = map_3[x] + Nside-x;
                  if (adjindex==3) {
                     fptr_real = fourierplane_real[HalfNside + x] + HalfNside -x;
                     fptr_imag = fourierplane_imag[HalfNside + x] + HalfNside -x;
                     fptr_real_inv = fourierplane_real[TwoNside-x] + TwoNside +x;
                     fptr_imag_inv = fourierplane_imag[TwoNside-x] + TwoNside +x;
                  } else {
                     fptr_real = fourierplane_real[OneAndHalfNside + x] + HalfNside -x;
                     fptr_imag = fourierplane_imag[OneAndHalfNside + x] + HalfNside -x;
                     fptr_real_inv = fourierplane_real[Nside-x] + TwoNside +x;
                     fptr_imag_inv = fourierplane_imag[Nside-x] + TwoNside +x;
                  }
                  ymax = Nside/2-x-1;
                  for(y=0;y<=ymax;y++) {
                     *fptr_real += *(map_ptr_0--);
                     *fptr_imag += *(map_ptr_1--);
                     *fptr_real_inv += *(map_ptr_2--);
                     *fptr_imag_inv += *(map_ptr_3--);
                     fptr_real--;
                     fptr_imag--;
                     fptr_real_inv++;
                     fptr_imag_inv++;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 6:
            case 8:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[x];
                  map_ptr_1 = map_1[x];
                  map_ptr_2 = map_2[x];
                  map_ptr_3 = map_3[x];
                  if (adjindex==6) {
                     fptr_real = fourierplane_real[OneAndHalfNside + x] + HalfNside;
                     fptr_imag = fourierplane_imag[OneAndHalfNside + x] + HalfNside;
                     fptr_real_inv = fourierplane_real[Nside-x] + TwoNside;
                     fptr_imag_inv = fourierplane_imag[Nside-x] + TwoNside;
                  } else {
                     fptr_real = fourierplane_real[HalfNside + x] + OneAndHalfNside;
                     fptr_imag = fourierplane_imag[HalfNside + x] + OneAndHalfNside;
                     fptr_real_inv = fourierplane_real[TwoNside-x] + Nside;
                     fptr_imag_inv = fourierplane_imag[TwoNside-x] + Nside;
                  }
                  ymax = Nside/2-x-1;
                  for(y=0;y<=ymax;y++) {
                     *fptr_real += *(map_ptr_0++);
                     *fptr_imag += *(map_ptr_1++);
                     *fptr_real_inv += *(map_ptr_2++);
                     *fptr_imag_inv += *(map_ptr_3++);
                     fptr_real++;
                     fptr_imag++;
                     fptr_real_inv--;
                     fptr_imag_inv--;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            default:
               fprintf(stderr, "Error in fftconv_icosa_1: You can't get here: switch(adjindex)-->default.\n");
               exit(1);
               break;

         } /* End switch(adjindex) */
      } /* End for(adjindex) */

#ifndef NOSMOOTH
      /* Smooth out the m-triangles by putting them into the p-triangles. */
      for(x=2;x<Nside/2;x++) {
         for(y=1;y<x;y++) {
            fracp = 1 - (fracm = 0.5 - 0.5 * (double)y/x);

            /* Triangle 2 */
            fourierplane_real[HalfNside-y][HalfNside-x+y] += fracp * fourierplane_real[HalfNside-x][HalfNside+y];
            fourierplane_imag[HalfNside-y][HalfNside-x+y] += fracp * fourierplane_imag[HalfNside-x][HalfNside+y];
            fourierplane_real[TwoNside +y][TwoNside +x-y] += fracp * fourierplane_real[TwoNside +x][TwoNside -y];
            fourierplane_imag[TwoNside +y][TwoNside +x-y] += fracp * fourierplane_imag[TwoNside +x][TwoNside -y];
            fourierplane_real[HalfNside-x][HalfNside+y] *= fracm;
            fourierplane_imag[HalfNside-x][HalfNside+y] *= fracm;
            fourierplane_real[TwoNside +x][TwoNside -y] *= fracm;
            fourierplane_imag[TwoNside +x][TwoNside -y] *= fracm;

            /* Triangle 3 */
            fourierplane_real[HalfNside-x+y][HalfNside-y] += fracp * fourierplane_real[HalfNside+y][HalfNside-x];
            fourierplane_imag[HalfNside-x+y][HalfNside-y] += fracp * fourierplane_imag[HalfNside+y][HalfNside-x];
            fourierplane_real[TwoNside +x-y][TwoNside +y] += fracp * fourierplane_real[TwoNside -y][TwoNside +x];
            fourierplane_imag[TwoNside +x-y][TwoNside +y] += fracp * fourierplane_imag[TwoNside -y][TwoNside +x];
            fourierplane_real[HalfNside+y][HalfNside-x] *= fracm;
            fourierplane_imag[HalfNside+y][HalfNside-x] *= fracm;
            fourierplane_real[TwoNside -y][TwoNside +x] *= fracm;
            fourierplane_imag[TwoNside -y][TwoNside +x] *= fracm;

            /* Triangle 5 */
            fourierplane_real[OneAndHalfNside+x][HalfNside-y] += fracp * fourierplane_real[OneAndHalfNside+x-y][HalfNside-x];
            fourierplane_imag[OneAndHalfNside+x][HalfNside-y] += fracp * fourierplane_imag[OneAndHalfNside+x-y][HalfNside-x];
            fourierplane_real[Nside-x][TwoNside+y] += fracp * fourierplane_real[Nside-x+y][TwoNside+x];
            fourierplane_imag[Nside-x][TwoNside+y] += fracp * fourierplane_imag[Nside-x+y][TwoNside+x];
            fourierplane_real[OneAndHalfNside+x-y][HalfNside-x] *= fracm;
            fourierplane_imag[OneAndHalfNside+x-y][HalfNside-x] *= fracm;
            fourierplane_real[Nside-x+y][TwoNside+x] *= fracm;
            fourierplane_imag[Nside-x+y][TwoNside+x] *= fracm;

            /* Triangle 6 */
            fourierplane_real[OneAndHalfNside+x][HalfNside-x+y] += fracp * fourierplane_real[OneAndHalfNside+x-y][HalfNside+y];
            fourierplane_imag[OneAndHalfNside+x][HalfNside-x+y] += fracp * fourierplane_imag[OneAndHalfNside+x-y][HalfNside+y];
            fourierplane_real[Nside-x][TwoNside+x-y] += fracp * fourierplane_real[Nside-x+y][TwoNside-y];
            fourierplane_imag[Nside-x][TwoNside+x-y] += fracp * fourierplane_imag[Nside-x+y][TwoNside-y];
            fourierplane_real[OneAndHalfNside+x-y][HalfNside+y] *= fracm;
            fourierplane_imag[OneAndHalfNside+x-y][HalfNside+y] *= fracm;
            fourierplane_real[Nside-x+y][TwoNside-y] *= fracm;
            fourierplane_imag[Nside-x+y][TwoNside-y] *= fracm;

            /* Triangle 8 */
            fourierplane_real[HalfNside-x+y][OneAndHalfNside+x] += fracp * fourierplane_real[HalfNside+y][OneAndHalfNside+x-y];
            fourierplane_imag[HalfNside-x+y][OneAndHalfNside+x] += fracp * fourierplane_imag[HalfNside+y][OneAndHalfNside+x-y];
            fourierplane_real[TwoNside+x-y][Nside-x] += fracp * fourierplane_real[TwoNside-y][Nside-x+y];
            fourierplane_imag[TwoNside+x-y][Nside-x] += fracp * fourierplane_imag[TwoNside-y][Nside-x+y];
            fourierplane_real[HalfNside+y][OneAndHalfNside+x-y] *= fracm;
            fourierplane_imag[HalfNside+y][OneAndHalfNside+x-y] *= fracm;
            fourierplane_real[TwoNside-y][Nside-x+y] *= fracm;
            fourierplane_imag[TwoNside-y][Nside-x+y] *= fracm;

            /* Triangle 9 */
            fourierplane_real[HalfNside-y][OneAndHalfNside+x] += fracp * fourierplane_real[HalfNside-x][OneAndHalfNside+x-y];
            fourierplane_imag[HalfNside-y][OneAndHalfNside+x] += fracp * fourierplane_imag[HalfNside-x][OneAndHalfNside+x-y];
            fourierplane_real[TwoNside+y][Nside-x] += fracp * fourierplane_real[TwoNside+x][Nside-x+y];
            fourierplane_imag[TwoNside+y][Nside-x] += fracp * fourierplane_imag[TwoNside+x][Nside-x+y];
            fourierplane_real[HalfNside-x][OneAndHalfNside+x-y] *= fracm;
            fourierplane_imag[HalfNside-x][OneAndHalfNside+x-y] *= fracm;
            fourierplane_real[TwoNside+x][Nside-x+y] *= fracm;
            fourierplane_imag[TwoNside+x][Nside-x+y] *= fracm;

         } /* End for(y) */
      } /* End for(x) */

      /* Edges */
      for(x=1;x<Nside/2;x++) {
         /* Triangles 2 & 3 */
         temp = 0.5*(fourierplane_real[HalfNside][HalfNside-x] + fourierplane_real[HalfNside-x][HalfNside]);
         fourierplane_real[HalfNside][HalfNside-x] = fourierplane_real[HalfNside-x][HalfNside] = temp;
         temp = 0.5*(fourierplane_imag[HalfNside][HalfNside-x] + fourierplane_imag[HalfNside-x][HalfNside]);
         fourierplane_imag[HalfNside][HalfNside-x] = fourierplane_imag[HalfNside-x][HalfNside] = temp;

         /* Triangles 5 & 6 */
         temp = 0.5*(fourierplane_real[OneAndHalfNside+x][HalfNside-x] + fourierplane_real[OneAndHalfNside+x][HalfNside]);
         fourierplane_real[OneAndHalfNside+x][HalfNside-x] = fourierplane_real[OneAndHalfNside+x][HalfNside] = temp;
         temp = 0.5*(fourierplane_imag[OneAndHalfNside+x][HalfNside-x] + fourierplane_imag[OneAndHalfNside+x][HalfNside]);
         fourierplane_imag[OneAndHalfNside+x][HalfNside-x] = fourierplane_imag[OneAndHalfNside+x][HalfNside] = temp;

         /* Triangles 8 & 9 */
         temp = 0.5*(fourierplane_real[HalfNside][OneAndHalfNside+x] + fourierplane_real[HalfNside-x][OneAndHalfNside+x]);
         fourierplane_real[HalfNside][OneAndHalfNside+x] = fourierplane_real[HalfNside-x][OneAndHalfNside+x] = temp;
         temp = 0.5*(fourierplane_imag[HalfNside][OneAndHalfNside+x] + fourierplane_imag[HalfNside-x][OneAndHalfNside+x]);
         fourierplane_imag[HalfNside][OneAndHalfNside+x] = fourierplane_imag[HalfNside-x][OneAndHalfNside+x] = temp;
      } /* End for(x) */
#endif

      /* Now that we have built the Fourier plane, perform a 2 dimensional FFT */
      fourier_2d_1(fftlen, fftlen, fourierplane_real, fourierplane_imag, 1);

      /* Multiply by the kernel */
      for(x=0;x<fftlen;x++) {
         fptr_real = fourierplane_real[x];
         fptr_imag = fourierplane_imag[x];
         kptr_real = kernel_real[x];
         kptr_imag = kernel_imag[x];
         for(y=0;y<fftlen;y++) {
            tempr = *fptr_real;
            tempi = *fptr_imag;
            *fptr_real = tempr* *(kptr_real  ) - tempi* *(kptr_imag  );
            *fptr_imag = tempi* *(kptr_real++) + tempr* *(kptr_imag++);
            fptr_real++;
            fptr_imag++;
         }
      } /* End for(x) */

      /* Back to real space */
      fourier_2d_1(fftlen, fftlen, fourierplane_real, fourierplane_imag, -1);

      /* Output the results into outdata.  "norm" is a multiplying factor so
       * that the inverse-FFT is really the inverse of the forward-FFT (it
       * otherwise is the inverse multiplied by the number of pixels).
       */
      norm = 1./(double)(fftlen*fftlen);
      for(x=0;x<=Nside;x++) {
         map_ptr_0 = outdata->data[face0][x];
         map_ptr_1 = outdata->data[face1][x];
         map_ptr_2 = outdata->data[face2][x];
         map_ptr_3 = outdata->data[face3][x];
         fptr_real = fourierplane_real[x + HalfNside] + HalfNside;
         fptr_imag = fourierplane_imag[x + HalfNside] + HalfNside;
         fptr_real_inv = fourierplane_real[TwoNside-x] + TwoNside;
         fptr_imag_inv = fourierplane_imag[TwoNside-x] + TwoNside;
         ymax = Nside-x;
         for(y=0;y<=ymax;y++) {
            *map_ptr_0 += norm* *(fptr_real++);
            *map_ptr_1 += norm* *(fptr_imag++);
            *map_ptr_2 += norm* *(fptr_real_inv--);
            *map_ptr_3 += norm* *(fptr_imag_inv--);
            map_ptr_0++;
            map_ptr_1++;
            map_ptr_2++;
            map_ptr_3++;
         } /* End for(y) */
      } /* End for(x) */

   } /* End for(fcounter) */

   /* Clean up temporary data structures */
   deallocate_icosahedral(&dataS1);
   deallocate_icosahedral(&dataS2);
   free_dmatrix(fourierplane_real, 0, fftlen-1, 0, fftlen-1);
   free_dmatrix(fourierplane_imag, 0, fftlen-1, 0, fftlen-1);
}

/* fftconv_icosa_rev_1
 * *** FFT CONVOLUTION ON AN ICOSAHEDRON ***
 *
 * Takes the given Fourier-space kernel (produced by conv_kernel_icosa_1) and
 * performs a flat-sky convolution on the icosahedral data indata.  The result
 * is added to outdata.  Nside must be even.  This is the transpose operation
 * of fftconv_icosa_1.
 *
 * Arguments:
 *   indata: input data
 * > outdata: output data
 *   fftlen: size of FFT plane (must be >2.5*Nside)
 *   kernel_real: real part of kernel
 *   kernel_imag: imag part of kernel
 */

void fftconv_icosa_rev_1(ICOSAHEDRAL *indata, ICOSAHEDRAL *outdata, long fftlen, double **kernel_real,
   double **kernel_imag) {

   int face, face0, face1, face2, face3, fcounter, adjindex;
   double **face_ptr, **face_ptrS1, **face_ptrS2;
   double tempr, tempi;
   long x,y,ymax;
   long Nside = indata->Nside;
   long HalfNside = Nside >> 1;
   long TwoNside = Nside << 1;
   long OneAndHalfNside = HalfNside+Nside;
   double norm;
   double fracm, fracp, temp;
   double **fourierplane_real, **fourierplane_imag;
   double *fptr_real, *fptr_imag, *fptr_real_inv, *fptr_imag_inv, *kptr_real, *kptr_imag;
   double *map_ptr_0, *map_ptr_1, *map_ptr_2, *map_ptr_3;
   double **map_0, **map_1, **map_2, **map_3;
   int *adjptr;
   char *rotadjptr;
   ICOSAHEDRAL dataS1, dataS2; /* rotated data */

   /* Table of adjacent faces */
   int adj1[] = { 5,  1,  2,  3,  4,  1,  2,  3,  4,  5, 10,  6,  7,  8,  9, 20, 16, 17, 18, 19};
   int adj2[] = {10,  6,  7,  8,  9,  5,  1,  2,  3,  4, 15, 11, 12, 13, 14, 19, 20, 16, 17, 18};
   int adj3[] = {11, 12, 13, 14, 15, 10,  6,  7,  8,  9, 17, 18, 19, 20, 16, 18, 19, 20, 16, 17};
   int adj4[] = { 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 16, 17, 17, 18, 19, 20, 16};
   int adj5[] = {12, 13, 14, 15, 11, 18, 19, 20, 16, 17, 19, 20, 16, 17, 18, 15, 11, 12, 13, 14};
   int adj6[] = { 7,  8,  9, 10,  6, 19, 20, 16, 17, 18, 12, 13, 14, 15, 11,  9, 10,  6,  7,  8};
   int adj7[] = { 2,  3,  4,  5,  1, 12, 13, 14, 15, 11,  6,  7,  8,  9, 10, 14, 15, 11, 12, 13};
   int adj8[] = { 3,  4,  5,  1,  2,  7,  8,  9, 10,  6,  1,  2,  3,  4,  5,  8,  9, 10,  6,  7};
   int adj9[] = { 4,  5,  1,  2,  3,  2,  3,  4,  5,  1,  5,  1,  2,  3,  4, 13, 14, 15, 11, 12};

   /* Table of orientations of adjacent faces, relative to ab=horizontal */
   char rotadj1[] = "aaaaacccccaaaaaccccc";
   char rotadj2[] = "cccccbbbbbbbbbbaaaaa";
   char rotadj3[] = "bbbbbbbbbbaaaaaccccc";
   char rotadj4[] = "aaaaaccccccccccaaaaa";
   char rotadj5[] = "bbbbbaaaaabbbbbccccc";
   char rotadj6[] = "bbbbbaaaaabbbbbccccc";
   char rotadj7[] = "ccccccccccaaaaaaaaaa";
   char rotadj8[] = "aaaaabbbbbbbbbbccccc";
   char rotadj9[] = "cccccaaaaabbbbbbbbbb";

   /* Pointers to the tables */
   int *adj[] = {NULL, adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, adj9};
   char *rotadj[] = {NULL, rotadj1, rotadj2, rotadj3, rotadj4, rotadj5, rotadj6, rotadj7, rotadj8, rotadj9};

#ifdef N_CHECKVAL
   /* Check for consistency of input and output array sizes */
   if (indata->Nside != outdata->Nside) {
      fprintf(stderr, "Error in fftconv_icosa_rev_1: input and output sizes (%ld vs. %ld) do not match.\n",
         indata->Nside, outdata->Nside);
      exit(1);
   }

   /* We need to test for fftlen being too small because this could cause wrap-around effects. */
   if (fftlen<<1<5*Nside) {
      fprintf(stderr, "Error in fftconv_icosa_rev_1: fftlen=%ld, Nside=%ld are illegal values.\n", fftlen, Nside);
      exit(1);
   }

   /* Check that Nside is even */
   if (Nside & 1) {
      fprintf(stderr, "Error in fftconv_icosa_rev_1: Nside=%ld is odd.\n", Nside);
      exit(1);
   }
#endif

   /* Build data structures rotated by 120 and 240 degrees */
   allocate_icosahedral(&dataS1, Nside);
   allocate_icosahedral(&dataS2, Nside);

   fourierplane_real = dmatrix(0, fftlen-1, 0, fftlen-1);
   fourierplane_imag = dmatrix(0, fftlen-1, 0, fftlen-1);

   /* Loop over faces.  The fcounter=0..4 will indicate which quadruplet of faces is being considered.
    * The individual faces are then face = 1..20 = fcounter*4 + findex + 1;
    */
   for(fcounter=0;fcounter<5;fcounter++) {

      /* Get face numbers */
      face0 = (fcounter<<2)+1;
      face1 = (fcounter<<2)+2;
      face2 = (fcounter<<2)+3;
      face3 = (fcounter<<2)+4;

      /* Clear the Fourier plane */
      for(x=0;x<fftlen;x++) {
         fptr_real = fourierplane_real[x];
         fptr_imag = fourierplane_imag[x];
         for(y=0;y<fftlen;y++)
            *(++fptr_real) = *(++fptr_imag) = 0.;
      } /* End for(x) */

      /* Read in the results from indata.  "norm" is a multiplying factor so
       * that the inverse-FFT is really the inverse of the forward-FFT (it
       * otherwise is the inverse multiplied by the number of pixels).
       */
      norm = 1./(double)(fftlen*fftlen);
      for(x=0;x<=Nside;x++) {
         map_ptr_0 = indata->data[face0][x];
         map_ptr_1 = indata->data[face1][x];
         map_ptr_2 = indata->data[face2][x];
         map_ptr_3 = indata->data[face3][x];
         fptr_real = fourierplane_real[x + HalfNside] + HalfNside;
         fptr_imag = fourierplane_imag[x + HalfNside] + HalfNside;
         fptr_real_inv = fourierplane_real[TwoNside-x] + TwoNside;
         fptr_imag_inv = fourierplane_imag[TwoNside-x] + TwoNside;
         ymax = Nside-x;
         for(y=0;y<=ymax;y++) {
            *fptr_real += norm* *(map_ptr_0++);
            *fptr_imag += norm* *(map_ptr_1++);
            *fptr_real_inv += norm* *(map_ptr_2++);
            *fptr_imag_inv += norm* *(map_ptr_3++);
            fptr_real++;
            fptr_imag++;
            fptr_real_inv--;
            fptr_imag_inv--;
         } /* End for(y) */
      } /* End for(x) */

      /* Real --> Fourier space (transpose of forward Fourier-->real operation) */
      fourier_2d_1(fftlen, fftlen, fourierplane_real, fourierplane_imag, 1);

      /* Multiply by the kernel */
      for(x=0;x<fftlen;x++) {
         fptr_real = fourierplane_real[x];
         fptr_imag = fourierplane_imag[x];
         kptr_real = kernel_real[x];
         kptr_imag = kernel_imag[x];
         for(y=0;y<fftlen;y++) {
            tempr = *fptr_real;
            tempi = *fptr_imag;
            *fptr_real = tempr* *(kptr_real  ) + tempi* *(kptr_imag  );
            *fptr_imag = tempi* *(kptr_real++) - tempr* *(kptr_imag++);
            fptr_real++;
            fptr_imag++;
         }
      } /* End for(x) */

      /* Fourier --> real space (transpose of forward real-->Fourier operation) */
      fourier_2d_1(fftlen, fftlen, fourierplane_real, fourierplane_imag, -1);

#ifndef NOSMOOTH
      /* Smooth out the m-triangles by putting them into the p-triangles. */
      for(x=2;x<Nside/2;x++) {
         for(y=1;y<x;y++) {
            fracp = 1 - (fracm = 0.5 - 0.5 * (double)y/x);

            /* Triangle 2 */
            fourierplane_real[HalfNside-x][HalfNside+y] *= fracm;
            fourierplane_imag[HalfNside-x][HalfNside+y] *= fracm;
            fourierplane_real[TwoNside +x][TwoNside -y] *= fracm;
            fourierplane_imag[TwoNside +x][TwoNside -y] *= fracm;
            fourierplane_real[HalfNside-x][HalfNside+y] += fracp * fourierplane_real[HalfNside-y][HalfNside-x+y];
            fourierplane_imag[HalfNside-x][HalfNside+y] += fracp * fourierplane_imag[HalfNside-y][HalfNside-x+y];
            fourierplane_real[TwoNside +x][TwoNside -y] += fracp * fourierplane_real[TwoNside +y][TwoNside +x-y];
            fourierplane_imag[TwoNside +x][TwoNside -y] += fracp * fourierplane_imag[TwoNside +y][TwoNside +x-y];

            /* Triangle 3 */
            fourierplane_real[HalfNside+y][HalfNside-x] *= fracm;
            fourierplane_imag[HalfNside+y][HalfNside-x] *= fracm;
            fourierplane_real[TwoNside -y][TwoNside +x] *= fracm;
            fourierplane_imag[TwoNside -y][TwoNside +x] *= fracm;
            fourierplane_real[HalfNside+y][HalfNside-x] += fracp * fourierplane_real[HalfNside-x+y][HalfNside-y];
            fourierplane_imag[HalfNside+y][HalfNside-x] += fracp * fourierplane_imag[HalfNside-x+y][HalfNside-y];
            fourierplane_real[TwoNside -y][TwoNside +x] += fracp * fourierplane_real[TwoNside +x-y][TwoNside +y];
            fourierplane_imag[TwoNside -y][TwoNside +x] += fracp * fourierplane_imag[TwoNside +x-y][TwoNside +y];

            /* Triangle 5 */
            fourierplane_real[OneAndHalfNside+x-y][HalfNside-x] *= fracm;
            fourierplane_imag[OneAndHalfNside+x-y][HalfNside-x] *= fracm;
            fourierplane_real[Nside-x+y][TwoNside+x] *= fracm;
            fourierplane_imag[Nside-x+y][TwoNside+x] *= fracm;
            fourierplane_real[OneAndHalfNside+x-y][HalfNside-x] += fracp * fourierplane_real[OneAndHalfNside+x][HalfNside-y];
            fourierplane_imag[OneAndHalfNside+x-y][HalfNside-x] += fracp * fourierplane_imag[OneAndHalfNside+x][HalfNside-y];
            fourierplane_real[Nside-x+y][TwoNside+x] += fracp * fourierplane_real[Nside-x][TwoNside+y];
            fourierplane_imag[Nside-x+y][TwoNside+x] += fracp * fourierplane_imag[Nside-x][TwoNside+y];

            /* Triangle 6 */
            fourierplane_real[OneAndHalfNside+x-y][HalfNside+y] *= fracm;
            fourierplane_imag[OneAndHalfNside+x-y][HalfNside+y] *= fracm;
            fourierplane_real[Nside-x+y][TwoNside-y] *= fracm;
            fourierplane_imag[Nside-x+y][TwoNside-y] *= fracm;
            fourierplane_real[OneAndHalfNside+x-y][HalfNside+y] += fracp * fourierplane_real[OneAndHalfNside+x][HalfNside-x+y];
            fourierplane_imag[OneAndHalfNside+x-y][HalfNside+y] += fracp * fourierplane_imag[OneAndHalfNside+x][HalfNside-x+y];
            fourierplane_real[Nside-x+y][TwoNside-y] += fracp * fourierplane_real[Nside-x][TwoNside+x-y];
            fourierplane_imag[Nside-x+y][TwoNside-y] += fracp * fourierplane_imag[Nside-x][TwoNside+x-y];

            /* Triangle 8 */
            fourierplane_real[HalfNside+y][OneAndHalfNside+x-y] *= fracm;
            fourierplane_imag[HalfNside+y][OneAndHalfNside+x-y] *= fracm;
            fourierplane_real[TwoNside-y][Nside-x+y] *= fracm;
            fourierplane_imag[TwoNside-y][Nside-x+y] *= fracm;
            fourierplane_real[HalfNside+y][OneAndHalfNside+x-y] += fracp * fourierplane_real[HalfNside-x+y][OneAndHalfNside+x];
            fourierplane_imag[HalfNside+y][OneAndHalfNside+x-y] += fracp * fourierplane_imag[HalfNside-x+y][OneAndHalfNside+x];
            fourierplane_real[TwoNside-y][Nside-x+y] += fracp * fourierplane_real[TwoNside+x-y][Nside-x];
            fourierplane_imag[TwoNside-y][Nside-x+y] += fracp * fourierplane_imag[TwoNside+x-y][Nside-x];

            /* Triangle 9 */
            fourierplane_real[HalfNside-x][OneAndHalfNside+x-y] *= fracm;
            fourierplane_imag[HalfNside-x][OneAndHalfNside+x-y] *= fracm;
            fourierplane_real[TwoNside+x][Nside-x+y] *= fracm;
            fourierplane_imag[TwoNside+x][Nside-x+y] *= fracm;
            fourierplane_real[HalfNside-x][OneAndHalfNside+x-y] += fracp * fourierplane_real[HalfNside-y][OneAndHalfNside+x];
            fourierplane_imag[HalfNside-x][OneAndHalfNside+x-y] += fracp * fourierplane_imag[HalfNside-y][OneAndHalfNside+x];
            fourierplane_real[TwoNside+x][Nside-x+y] += fracp * fourierplane_real[TwoNside+y][Nside-x];
            fourierplane_imag[TwoNside+x][Nside-x+y] += fracp * fourierplane_imag[TwoNside+y][Nside-x];

         } /* End for(y) */
      } /* End for(x) */

      /* Edges */
      for(x=1;x<Nside/2;x++) {
         /* Triangles 2 & 3 */
         temp = 0.5*(fourierplane_real[HalfNside][HalfNside-x] + fourierplane_real[HalfNside-x][HalfNside]);
         fourierplane_real[HalfNside][HalfNside-x] = fourierplane_real[HalfNside-x][HalfNside] = temp;
         temp = 0.5*(fourierplane_imag[HalfNside][HalfNside-x] + fourierplane_imag[HalfNside-x][HalfNside]);
         fourierplane_imag[HalfNside][HalfNside-x] = fourierplane_imag[HalfNside-x][HalfNside] = temp;

         /* Triangles 5 & 6 */
         temp = 0.5*(fourierplane_real[OneAndHalfNside+x][HalfNside-x] + fourierplane_real[OneAndHalfNside+x][HalfNside]);
         fourierplane_real[OneAndHalfNside+x][HalfNside-x] = fourierplane_real[OneAndHalfNside+x][HalfNside] = temp;
         temp = 0.5*(fourierplane_imag[OneAndHalfNside+x][HalfNside-x] + fourierplane_imag[OneAndHalfNside+x][HalfNside]);
         fourierplane_imag[OneAndHalfNside+x][HalfNside-x] = fourierplane_imag[OneAndHalfNside+x][HalfNside] = temp;

         /* Triangles 8 & 9 */
         temp = 0.5*(fourierplane_real[HalfNside][OneAndHalfNside+x] + fourierplane_real[HalfNside-x][OneAndHalfNside+x]);
         fourierplane_real[HalfNside][OneAndHalfNside+x] = fourierplane_real[HalfNside-x][OneAndHalfNside+x] = temp;
         temp = 0.5*(fourierplane_imag[HalfNside][OneAndHalfNside+x] + fourierplane_imag[HalfNside-x][OneAndHalfNside+x]);
         fourierplane_imag[HalfNside][OneAndHalfNside+x] = fourierplane_imag[HalfNside-x][OneAndHalfNside+x] = temp;
      } /* End for(x) */
#endif

      /* Build the central triangle for each findex.  This includes setting the face
       * numbers face0 .. face3 of the triangles we're going to compute.
       */
      for(x=0;x<=Nside;x++) {
         map_ptr_0 = outdata->data[face0][x];
         map_ptr_1 = outdata->data[face1][x];
         map_ptr_2 = outdata->data[face2][x];
         map_ptr_3 = outdata->data[face3][x];
         fptr_real = fourierplane_real[x + HalfNside] + HalfNside;
         fptr_imag = fourierplane_imag[x + HalfNside] + HalfNside;
         fptr_real_inv = fourierplane_real[TwoNside-x] + TwoNside;
         fptr_imag_inv = fourierplane_imag[TwoNside-x] + TwoNside;
         ymax = Nside-x;
         for(y=0;y<=ymax;y++) {
            *map_ptr_0 += *(fptr_real++);
            *map_ptr_1 += *(fptr_imag++);
            *map_ptr_2 += *(fptr_real_inv--);
            *map_ptr_3 += *(fptr_imag_inv--);
            map_ptr_0++;
            map_ptr_1++;
            map_ptr_2++;
            map_ptr_3++;
         } /* End for(y) */
      } /* End for(x) */

      /* Now build the adjacent faces */
      for(adjindex=1; adjindex<=9; adjindex++) {

         /* Identify which adjacent face we want, and in which orientation */
         adjptr = adj[adjindex];
         rotadjptr = rotadj[adjindex];
         map_0 = rotadj[adjindex][face0-1]=='a'?  dataS1.data[adj[adjindex][face0-1]]:
                 rotadj[adjindex][face0-1]=='b'? outdata->data[adj[adjindex][face0-1]]:  dataS2.data[adj[adjindex][face0-1]];
         map_1 = rotadj[adjindex][face1-1]=='a'?  dataS1.data[adj[adjindex][face1-1]]:
                 rotadj[adjindex][face1-1]=='b'? outdata->data[adj[adjindex][face1-1]]:  dataS2.data[adj[adjindex][face1-1]];
         map_2 = rotadj[adjindex][face2-1]=='a'?  dataS1.data[adj[adjindex][face2-1]]:
                 rotadj[adjindex][face2-1]=='b'? outdata->data[adj[adjindex][face2-1]]:  dataS2.data[adj[adjindex][face2-1]];
         map_3 = rotadj[adjindex][face3-1]=='a'?  dataS1.data[adj[adjindex][face3-1]]:
                 rotadj[adjindex][face3-1]=='b'? outdata->data[adj[adjindex][face3-1]]:  dataS2.data[adj[adjindex][face3-1]];

         switch(adjindex) {
            case 1:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[x];
                  map_ptr_1 = map_1[x];
                  map_ptr_2 = map_2[x];
                  map_ptr_3 = map_3[x];
                  fptr_real = fourierplane_real[HalfNside - x] + OneAndHalfNside;
                  fptr_imag = fourierplane_imag[HalfNside - x] + OneAndHalfNside;
                  fptr_real_inv = fourierplane_real[TwoNside+x] + Nside;
                  fptr_imag_inv = fourierplane_imag[TwoNside+x] + Nside;
                  ymax = Nside-x;
                  for(y=0;y<=ymax;y++) {
                     *map_ptr_0 += *(fptr_real--);
                     *map_ptr_1 += *(fptr_imag--);
                     *map_ptr_2 += *(fptr_real_inv++);
                     *map_ptr_3 += *(fptr_imag_inv++);
                     map_ptr_0++;
                     map_ptr_1++;
                     map_ptr_2++;
                     map_ptr_3++;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 4:
               for(x=0;x<=Nside;x++) {
                  map_ptr_0 = map_0[x];
                  map_ptr_1 = map_1[x];
                  map_ptr_2 = map_2[x];
                  map_ptr_3 = map_3[x];
                  fptr_real = fourierplane_real[OneAndHalfNside - x] + HalfNside;
                  fptr_imag = fourierplane_imag[OneAndHalfNside - x] + HalfNside;
                  fptr_real_inv = fourierplane_real[Nside+x] + TwoNside;
                  fptr_imag_inv = fourierplane_imag[Nside+x] + TwoNside;
                  ymax = x>HalfNside? Nside-x: HalfNside-1;
                  for(y=0;y<=ymax;y++) {
                     *map_ptr_0 += *(fptr_real--);
                     *map_ptr_1 += *(fptr_imag--);
                     *map_ptr_2 += *(fptr_real_inv++);
                     *map_ptr_3 += *(fptr_imag_inv++);
                     map_ptr_0++;
                     map_ptr_1++;
                     map_ptr_2++;
                     map_ptr_3++;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 7:
               for(x=0;x<=Nside;x++) {
                  map_ptr_0 = map_0[x] + Nside-x; /* Must resport to some trickery since this face */
                  map_ptr_1 = map_1[x] + Nside-x; /* is upside-down.                               */
                  map_ptr_2 = map_2[x] + Nside-x;
                  map_ptr_3 = map_3[x] + Nside-x;
                  fptr_real = fourierplane_real[OneAndHalfNside-x] + HalfNside+x;
                  fptr_imag = fourierplane_imag[OneAndHalfNside-x] + HalfNside+x;
                  fptr_real_inv = fourierplane_real[Nside+x] + TwoNside-x;
                  fptr_imag_inv = fourierplane_imag[Nside+x] + TwoNside-x;
                  ymax = x>HalfNside? Nside-x: HalfNside-1;
                  for(y=0;y<=ymax;y++) {
                     *map_ptr_0 += *(fptr_real++);
                     *map_ptr_1 += *(fptr_imag++);
                     *map_ptr_2 += *(fptr_real_inv--);
                     *map_ptr_3 += *(fptr_imag_inv--);
                     map_ptr_0--;
                     map_ptr_1--;
                     map_ptr_2--;
                     map_ptr_3--;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 2:
            case 9:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[Nside-x]; /* note this is reflected */
                  map_ptr_1 = map_1[Nside-x];
                  map_ptr_2 = map_2[Nside-x];
                  map_ptr_3 = map_3[Nside-x];
                  fptr_real = fourierplane_real[HalfNside - x] + HalfNside;
                  fptr_imag = fourierplane_imag[HalfNside - x] + HalfNside;
                  fptr_real_inv = fourierplane_real[TwoNside+x] + TwoNside;
                  fptr_imag_inv = fourierplane_imag[TwoNside+x] + TwoNside;
                  if (adjindex==9) {
                     fptr_real += Nside;
                     fptr_imag += Nside;
                     fptr_real_inv -= Nside;
                     fptr_imag_inv -= Nside;
                  }
                  for(y=0;y<=x;y++) {
                     *map_ptr_0 += *(fptr_real++);
                     *map_ptr_1 += *(fptr_imag++);
                     *map_ptr_2 += *(fptr_real_inv--);
                     *map_ptr_3 += *(fptr_imag_inv--);
                     map_ptr_0++;
                     map_ptr_1++;
                     map_ptr_2++;
                     map_ptr_3++;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 3:
            case 5:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[x] + Nside-x; /* Go from "top" (highest y) down */
                  map_ptr_1 = map_1[x] + Nside-x;
                  map_ptr_2 = map_2[x] + Nside-x;
                  map_ptr_3 = map_3[x] + Nside-x;
                  if (adjindex==3) {
                     fptr_real = fourierplane_real[HalfNside + x] + HalfNside -x;
                     fptr_imag = fourierplane_imag[HalfNside + x] + HalfNside -x;
                     fptr_real_inv = fourierplane_real[TwoNside-x] + TwoNside +x;
                     fptr_imag_inv = fourierplane_imag[TwoNside-x] + TwoNside +x;
                  } else {
                     fptr_real = fourierplane_real[OneAndHalfNside + x] + HalfNside -x;
                     fptr_imag = fourierplane_imag[OneAndHalfNside + x] + HalfNside -x;
                     fptr_real_inv = fourierplane_real[Nside-x] + TwoNside +x;
                     fptr_imag_inv = fourierplane_imag[Nside-x] + TwoNside +x;
                  }
                  ymax = Nside/2-x-1;
                  for(y=0;y<=ymax;y++) {
                     *map_ptr_0 += *(fptr_real--);
                     *map_ptr_1 += *(fptr_imag--);
                     *map_ptr_2 += *(fptr_real_inv++);
                     *map_ptr_3 += *(fptr_imag_inv++);
                     map_ptr_0--;
                     map_ptr_1--;
                     map_ptr_2--;
                     map_ptr_3--;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            case 6:
            case 8:
               for(x=0;x<Nside/2;x++) {
                  map_ptr_0 = map_0[x];
                  map_ptr_1 = map_1[x];
                  map_ptr_2 = map_2[x];
                  map_ptr_3 = map_3[x];
                  if (adjindex==6) {
                     fptr_real = fourierplane_real[OneAndHalfNside + x] + HalfNside;
                     fptr_imag = fourierplane_imag[OneAndHalfNside + x] + HalfNside;
                     fptr_real_inv = fourierplane_real[Nside-x] + TwoNside;
                     fptr_imag_inv = fourierplane_imag[Nside-x] + TwoNside;
                  } else {
                     fptr_real = fourierplane_real[HalfNside + x] + OneAndHalfNside;
                     fptr_imag = fourierplane_imag[HalfNside + x] + OneAndHalfNside;
                     fptr_real_inv = fourierplane_real[TwoNside-x] + Nside;
                     fptr_imag_inv = fourierplane_imag[TwoNside-x] + Nside;
                  }
                  ymax = Nside/2-x-1;
                  for(y=0;y<=ymax;y++) {
                     *map_ptr_0 += *(fptr_real++);
                     *map_ptr_1 += *(fptr_imag++);
                     *map_ptr_2 += *(fptr_real_inv--);
                     *map_ptr_3 += *(fptr_imag_inv--);
                     map_ptr_0++;
                     map_ptr_1++;
                     map_ptr_2++;
                     map_ptr_3++;
                  } /* End for(y) */
               } /* End for(x) */
               break;

            default:
               fprintf(stderr, "Error in fftconv_icosa_rev_1: You can't get here: switch(adjindex)-->default.\n");
               exit(1);
               break;

         } /* End switch(adjindex) */
      } /* End for(adjindex) */

   } /* End for(fcounter) */

   /* Load into outdata the stuff that was placed in rotated triangles */
   for(face=1;face<=20;face++) {
      face_ptr = outdata->data[face];
      face_ptrS1 = dataS1.data[face];
      face_ptrS2 = dataS2.data[face];
      for(x=0;x<=Nside;x++)
         for(y=0;x+y<=Nside;y++)
            face_ptr[x][y] += face_ptrS1[Nside-x-y][x] + face_ptrS2[y][Nside-x-y];
   } /* End for(face,x,y) */

   /* Clean up temporary data structures */
   deallocate_icosahedral(&dataS1);
   deallocate_icosahedral(&dataS2);
   free_dmatrix(fourierplane_real, 0, fftlen-1, 0, fftlen-1);
   free_dmatrix(fourierplane_imag, 0, fftlen-1, 0, fftlen-1);
}

/* fftconv_icosa_2
 * *** CONVOLUTION ON ICOSAHEDRAL GRID ***
 *
 * Does a "spherical" convolution on the icosahedral grid using FFT at high L
 * and regular SHT at low L (with PM-interpolation).
 *
 * Arguments:
 *   indata: input data
 * > outdata: output data
 *   fftlen: size of FFT plane (must be >2.5*Nside)
 *   kernel_real: real part of kernel
 *   kernel_imag: imag part of kernel
 *   Cl: convolution kernel
 *   lbrutemax: maximum value of L to compute by brute force
 *   Icosahedron: spherical pixelization containing coordinates of data
 */

void fftconv_icosa_2(ICOSAHEDRAL *indata, ICOSAHEDRAL *outdata, long fftlen, double **kernel_real,
   double **kernel_imag, double *Cl, long lbrutemax, SPHERE_PIXEL *Icosahedron) {

   ICOSAHEDRAL intermediate_data;
   long nspace, i, L;
   double psi;
   double *outptr, *intptr, *Cl_cutoff;

#ifdef N_CHECKVAL
   if (indata->Npix != Icosahedron->N || outdata->Npix != Icosahedron->N) {
      fprintf(stderr, "Error: size mismatch in fftconv_icosa_2: %ld --> %ld (pixelization %ld).\n",
         indata->Npix, outdata->Npix, Icosahedron->N);
      exit(1);
   }

   if (lbrutemax <= ICOSA_DL) {
      fprintf(stderr, "Error in fftconv_icosa_2: must have lbrutemax=%ld>%d, or increase ICOSA_DL\n", lbrutemax, ICOSA_DL);
      exit(1);
   }
#endif

   /* Do the low-L stuff */
   nspace = 1;
   while (nspace<lbrutemax) nspace<<=1;
   nspace <<= 2;
   Cl_cutoff = dvector(0, lbrutemax);
   for(L=0;L<=lbrutemax;L++) Cl_cutoff[L] = Cl[L];
   for(L=lbrutemax - ICOSA_DL; L<=lbrutemax; L++) {
      psi = Pi * (lbrutemax - L) / (double) ICOSA_DL;
      Cl_cutoff[L] *= 0.5*(1-cos(psi));
   }
   pixel_convolution_2(indata->vector, NULL, outdata->vector, NULL, lbrutemax, Icosahedron, Icosahedron,
      Cl_cutoff, NULL, 0, 0, 0x1, nspace, 2);
   free_dvector(Cl_cutoff, 0, lbrutemax);

   /* Now for high-L stuff */
   allocate_icosahedral(&intermediate_data, indata->Nside);
   fftconv_icosa_1(indata, &intermediate_data, fftlen, kernel_real, kernel_imag);
   fftconv_icosa_rev_1(indata, &intermediate_data, fftlen, kernel_real, kernel_imag);
   outptr = outdata->vector;
   intptr = intermediate_data.vector;
   for(i=0;i<Icosahedron->N;i++) {
      *outptr += 0.5 * *(intptr++);
      outptr++;
   }
   deallocate_icosahedral(&intermediate_data);
}

/* inverse_fftconv_icosa_1
 * *** DE-CONVOLUTION ON ICOSAHEDRAL GRID ***
 *
 * Does an inverse "spherical" convolution on the icosahedral grid using FFT at high L
 * and regular SHT at low L (with PM-interpolation).  This routine uses the Woodbury
 * pre-conditioner and a Chebyshev polynomial C^{-1} method.
 *
 * The OpSelect codes are:
 *   0: do an approximate C^{-1}X
 *   1: recompute transfer matrix for given power spectrum Cl and lsplit
 *   2: write transfer matrix to a file
 *   3: read transfer matrix from a file
 *   4: clean up memory
 *
 * Arguments:
 *   indata: input data
 * > outdata: output data
 *   fftlen: size of FFT plane (must be >2.5*Nside)
 *   ifd: inverse-FFT data structure
 *   Cl: convolution kernel
 *   lsplit: maximum value of L to compute via Woodbury
 *   lbrutemax: maximum value of L to compute by brute force
 *   Icosahedron: spherical pixelization containing coordinates of data
 *   OpSelect: a flag telling the routine which operations to perform (see above)
 *   tnum: Woodbury preconditioner number
 *   FileName: name of file to read/write
 *   Noise: noise variances of the icosahedral pixels
 *   chebyshev_order: order of Chebyshev polynomial to use in preconditioner
 */
#ifndef THREAD_SAFE_SHT
void inverse_fftconv_icosa_1(ICOSAHEDRAL *indata, ICOSAHEDRAL *outdata, long fftlen, IFFT_DATA *ifd,
   double *Cl, long lsplit, long lbrutemax, SPHERE_PIXEL *Icosahedron, int OpSelect, int tnum,
   char FileName[], double *Noise, long chebyshev_order) {

#define CONDITION_ITER_FWD 16
#define UPPER_SAFETY_FACTOR 1.2

   FILE *fp;
   long i,j, n, Nside, x, y;
   static long Nicosa[TNUM_MAX+1];
   ICOSAHEDRAL ui, vi, wi, Avi;
   double *u, *v, *w, *Av;
   double clsplit;
   double temp, sqnorm, rho, dlam, tn_rho, tnm_rho, tnp_rho, lam0inv;
   double alpha, beta, gamma;
   static double condition_upper[TNUM_MAX+1];
   static double condition_lower[TNUM_MAX+1];
   char FileNameExt[1024];

   /* Initialize to avoid innocuous but annoying compiler warnings */
   u = v = w = Av = NULL;

   switch(OpSelect) {

      case 0: /* Perform a preconditioning multiplication */

         /* The method is as follows: construct the polynomial f_n(lambda) by
          *
          * lambda f_n(lambda) = 1 - T_n((lambda_0 - lambda) / Delta lambda) / T_n(lambda_0/Delta lambda).
          *
          * Here lambda_0 = (lambda_max + lambda_min)/2 and Delta lambda = (lambda_max - lambda_min)/2
          * where lambda_min and lambda_max are the eigenvalue ranges of C(Woodbury)^-1 * C(FFT), and
          * T_n is a Chebyshev polynomial.  Define rho = lambda_0/Delta lambda.
          *
          * We then want to implement the operation
          *
          * u_n = f_n[C(FFT) C(Woodbury)^-1] * X,    y_n = C(Woodbury)^-1 * u_n,
          *
          * as a preconditioner (approximate C(FFT)^-1 * X).
          *
          * The trick is to use the recursion relation,
          *
          * u_{n+1} = 2 rho T_n(rho) u_n - (T_n(rho)/(Delta lambda * T_{n+1}(rho))) * ( -x + C(FFT) C(Woodbury)^-1 u_n )
          *   - T_{n-1}(rho)/T_{n+1}(rho) * u_{n-1}.
          */

#ifdef N_CHECKVAL
         if (chebyshev_order<2) {
            fprintf(stderr, "Error in inverse_fftconv_icosa_1: chebyshev_order=%ld should be >=2.\n", chebyshev_order);
         }
#endif

         rho = (condition_upper[tnum] + condition_lower[tnum])/(condition_upper[tnum] - condition_lower[tnum]);
         dlam = (condition_upper[tnum]-condition_lower[tnum]) / 2.0;
         lam0inv = 2.0 / (condition_upper[tnum]+condition_lower[tnum]);

         /* Figure out Nside */
         Nside = (long)floor(sqrt(Icosahedron->N / 10)) - 1;

         /* Initialize: u_0 = 0, u_1 = lambda_0^-1 * X */
         allocate_icosahedral(&ui, Nside);
         allocate_icosahedral(&vi, Nside);
         allocate_icosahedral(&wi, Nside);
         allocate_icosahedral(&Avi, Nside);
         u = ui.vector; v = vi.vector; w = wi.vector; Av = Avi.vector;

         for(i=0;i<Icosahedron->N;i++) {
            v[i] = 0.0; /* u0 */
            u[i] = lam0inv * indata->vector[i]; /* u1 */
         }

         /* Recursion time! */
         tnm_rho = 1;
         tn_rho = rho;
         for(n=1;n<chebyshev_order;n++) {

            /* Get coefficients */
            tnp_rho = 2.0 * rho * tn_rho - tnm_rho;
            alpha = 2.0 * rho * tn_rho / tnp_rho;
            beta = -2.0 * tn_rho / tnp_rho / dlam;
            gamma = -tnm_rho / tnp_rho;

            for(i=0;i<Icosahedron->N;i++) {
               Av[i] = -indata->vector[i];
               w[i] = 0.0;
            }
            woodbury_preconditioner_1(u, w, Icosahedron->theta, Icosahedron->phi, NULL, ifd[tnum].area_eff,
               Icosahedron->N, Cl, lsplit, FileName, 0, tnum);
            fftconv_icosa_2(&wi, &Avi, fftlen, ifd[tnum].kernel_real, ifd[tnum].kernel_imag, Cl, lbrutemax, Icosahedron);
            for(i=0;i<Icosahedron->N;i++) {
               Av[i] += Noise[i] * w[i];
               Av[i] = alpha*u[i] + beta*Av[i] + gamma*v[i];
               v[i] = u[i];
               u[i] = Av[i];  /* We have computed u[i] = u_n+1, v[i] = u_n */
            }

            /* Move Chebyshev polynomials */
            tnm_rho = tn_rho;
            tn_rho = tnp_rho;
         }

         /* Apply Woodbury to get final result */
         woodbury_preconditioner_1(u, outdata->vector, Icosahedron->theta, Icosahedron->phi, NULL, ifd[tnum].area_eff,
            Icosahedron->N, Cl, lsplit, FileName, 0, tnum);

         /* Cleanup */
         deallocate_icosahedral(&ui);
         deallocate_icosahedral(&vi);
         deallocate_icosahedral(&wi);
         deallocate_icosahedral(&Avi);
         break;

      case 1: /* Build preconditioner */

         /* Store icosahedral grid size */
         Nicosa[tnum] = Icosahedron->N;

         /* Figure out Nside */
         Nside = (long)floor(sqrt(Icosahedron->N / 10)) - 1;

         /* First build Woodbury */
         clsplit = Cl[lsplit];
         ifd[tnum].area_eff = dvector(0, Icosahedron->N-1);
         for(i=0;i<Icosahedron->N;i++) ifd[tnum].area_eff[i] = 1./(1./Icosahedron->area[i] + Noise[i]/clsplit);
         woodbury_preconditioner_1(NULL, NULL, Icosahedron->theta, Icosahedron->phi, NULL, ifd[tnum].area_eff,
            Icosahedron->N, Cl, lsplit, FileName, 1, tnum);

         /* Test for conditioning ratio by acting with C(Woodbury)^-1 * C(FFT) repeatedly on
          * a pseudo-random vector.  The latter we construct out of a pseudo-random sequence.
          */
         allocate_icosahedral(&ui, Nside);
         allocate_icosahedral(&vi, Nside);
         allocate_icosahedral(&Avi, Nside);
         u = ui.vector; v = vi.vector; Av = Avi.vector;

         v[0] = TwoCos72Deg;
         for(i=0;i<Icosahedron->N;i++) {
            temp = v[i-1] + TwoCos72Deg;
            v[i] = temp<1? temp: temp-1;
         }
         woodbury_preconditioner_1(v, Av, Icosahedron->theta, Icosahedron->phi, NULL, ifd[tnum].area_eff,
            Icosahedron->N, Cl, lsplit, FileName, 0, tnum);
         for(j=0;j<CONDITION_ITER_FWD;j++) {
            sqnorm=0;
            for(i=0;i<Icosahedron->N;i++) sqnorm += v[i]*Av[i];
            sqnorm = sqrt(sqnorm);

printf("#%3ld#%lg\n", j, sqnorm);

            for(i=0;i<Icosahedron->N;i++) { /* Re-normalize vector v so that v*C(Woodbury)^-1*v = 1 */
               v[i] = v[i]/sqnorm;
               Av[i] = Av[i]/sqnorm;
               u[i] = 0.0;
            }
            /* At this point, v is a vector and Av is C(Woodbury)^-1 v */

            fftconv_icosa_2(&Avi, &ui, fftlen, ifd[tnum].kernel_real, ifd[tnum].kernel_imag, Cl, lbrutemax, Icosahedron);
            for(i=0;i<Icosahedron->N;i++) {
               u[i] += Noise[i] * Av[i];
               v[i] = u[i];
               Av[i] = 0.0;
            }
            woodbury_preconditioner_1(v, Av, Icosahedron->theta, Icosahedron->phi, NULL, ifd[tnum].area_eff,
               Icosahedron->N, Cl, lsplit, FileName, 0, tnum);
         }
         condition_upper[tnum] = sqnorm * UPPER_SAFETY_FACTOR;
         condition_lower[tnum] = 1.5 * condition_upper[tnum] / (chebyshev_order * chebyshev_order);

         /* Cleanup */
         deallocate_icosahedral(&ui);
         deallocate_icosahedral(&vi);
         deallocate_icosahedral(&Avi);
         break;

      case 2: /* Write preconditioner.  There are two files, FileName and FileNameExt */

         woodbury_preconditioner_1(v, Av, NULL, NULL, NULL, ifd[tnum].area_eff,
               Nicosa[tnum], Cl, lsplit, FileName, 2, tnum);
         sprintf(FileNameExt, "%s.ext", FileName);
         fp = fopen(FileNameExt, "w");
         fprintf(fp, "%7ld %23.16lE %23.16lE\n", Nicosa[tnum], condition_lower[tnum], condition_upper[tnum]);
         for(i=0; i<Nicosa[tnum]; i++) fprintf(fp, "%23.16lE\n", ifd[tnum].area_eff[i]);
         for(x=0; x<fftlen; x++) for(y=0; y<fftlen; y++)
            fprintf(fp, "%23.16lE %23.16lE\n", ifd[tnum].kernel_real[x][y], ifd[tnum].kernel_imag[x][y]);
         fclose(fp);
         break;

      case 3: /* Read a preconditioner from disk */

         woodbury_preconditioner_1(v, Av, NULL, NULL, NULL, ifd[tnum].area_eff,
               0, Cl, lsplit, FileName, 3, tnum);
         sprintf(FileNameExt, "%s.ext", FileName);
         fp = fopen(FileNameExt, "r");
         fscanf(fp, "%ld %lg %lg", Nicosa+tnum, condition_lower+tnum, condition_upper+tnum);
         ifd[tnum].area_eff = dvector(0, Nicosa[tnum]-1);
         for(i=0; i<Nicosa[tnum]; i++) fscanf(fp, "%lg", ifd[tnum].area_eff+i);
         for(x=0; x<fftlen; x++) for(y=0; y<fftlen; y++)
            fscanf(fp, "%lg %lg", ifd[tnum].kernel_real[x]+y, ifd[tnum].kernel_imag[x]+y);
         fclose(fp);
         break;

      case 4: /* Kill preconditioner */

         woodbury_preconditioner_1(NULL, NULL, Icosahedron->theta, Icosahedron->phi, NULL, ifd[tnum].area_eff,
            Icosahedron->N, Cl, lsplit, FileName, 4, tnum);
         free_dvector(ifd[tnum].area_eff, 0, Icosahedron->N-1);
         break;

      default:
         fprintf(stderr, "Error: illegal OpSelect=%d in inverse_fftconv_icosa_1\n", OpSelect);
         exit(1);
         break;

   }
#undef CONDITION_ITER_FWD
#undef UPPER_SAFETY_FACTOR

}
#endif

/* inverse_fftconv_icosa_2
 * *** DE-CONVOLUTION PRECONDITIONER BASED ON ICOSAHEDRON ***
 *
 * Does an inverse "spherical" convolution on the icosahedral grid using FFT at high L
 * and regular SHT at low L (with PM-interpolation).  This routine uses
 * inverse_fftconv_icosa_1 as its kernel.
 *
 * The OpSelect codes are:
 *   0: do an approximate C^{-1}X
 *   1: recompute transfer matrix for given power spectrum Cl and lsplit
 *   2: write transfer matrix to a file
 *   3: read transfer matrix from a file
 *   4: clean up memory
 *
 * This code works using the following identity,
 *
 * (N+PSP')^-1 = N^-1 + N^-1 P V [(S+V)^-1 - V^-1] V P' N^-1
 *
 * where V = P' N^-1 P, and S and N are symmetric.  Our application is that N is the
 * noise matrix for the pixels, S is the signal for super-pixels (icosahedron), and P
 * is the projection matrix from the pixel to super-pixel space.  The (S+V)^-1 operation
 * is approximated using inverse_fftconv_icosa_1.
 *
 * Arguments:
 *   indata: input data
 * > outdata: output data
 *   theta: colatitudes of points
 *   phi: longitudes of points
 *   noisevar: noise variances of the icosahedral pixels
 *   Npix: number of pixels
 *   Cl: convolution kernel
 *   lsplit: maximum value of L to compute via Woodbury
 *   lbrutemax: maximum value of L to compute by brute force
 *   lfftmax: maximum value of L to compute by FFT
 *   Nside: nside for icosahedron
 *   chebyshev_order: order of Chebyshev polynomial to use in preconditioner
 *   FileName: name of file to read/write
 *   OpSelect: a flag telling the routine which operations to perform (see above)
 *   tnum: Woodbury preconditioner number
 */
#ifndef THREAD_SAFE_SHT
void inverse_fftconv_icosa_2(double *indata, double *outdata, double *theta, double *phi,
   double *noisevar, long Npix, double *Cl, long lsplit, long lbrutemax, long lfftmax, long Nside,
   long chebyshev_order, char FileName[], int OpSelect, int tnum) {

#define NFACTOR 100

   SPHERE_PIXEL Icosahedron;
   ICOSAHEDRAL DataX, DataCinvX;
   static IFFT_DATA ifd[TNUM_MAX+1];
   double *NoiseIcosahedron;
   double ivar_min;
   long L, i, fftlen;
   long *icosapix;

   /* fftlen must be >2.5*Nside and is a power of 2 because of its use in FFT. */
   fftlen = 8;
   while (fftlen < 5*(Nside>>1) + 4) fftlen<<=1;

   /* For read, write, or clear, we just need to tell inverse_fftconv_icosa_1 what to do. */
   if (OpSelect==2 || OpSelect==3 || OpSelect==4) {
      make_icosa_grid_1(&Icosahedron, Nside);
      inverse_fftconv_icosa_1(NULL, NULL, fftlen, ifd, Cl, lsplit, lbrutemax, &Icosahedron,
         OpSelect, tnum, FileName, NULL, 0);
      deallocate_sphere_pixel(&Icosahedron);
      return;
   }

   /* Options 0 and 1 (run and setup) require us to do the operations below. */

   /* Setup -- generate icosahedral grid */
   make_icosa_grid_1(&Icosahedron, Nside);
   if (OpSelect == 1) {
      ifd[tnum].kernel_real = dmatrix(0,fftlen-1,0,fftlen-1);
      ifd[tnum].kernel_imag = dmatrix(0,fftlen-1,0,fftlen-1);
      kernel_icosa_1(Nside, fftlen, lbrutemax, lfftmax, Cl, ifd[tnum].kernel_real, ifd[tnum].kernel_imag);
   }
   allocate_icosahedral(&DataX, Nside);
   allocate_icosahedral(&DataCinvX, Nside);
   NoiseIcosahedron = dvector(0, Icosahedron.N-1);
   icosapix = lvector(0, Npix-1);

   /* Find minimum allowable inverse noise.  This noise will be NFACTOR times the
    * maximum Cl per pixel.
    */
   ivar_min = 0;
   for(L=0;L<=lfftmax;L++) if (ivar_min<Cl[L]) ivar_min = Cl[L];
   ivar_min = FourPi/(NFACTOR * ivar_min * Icosahedron.N);

   /* Construct data and noise map on icosahedral grid */
   for(i=0; i<Icosahedron.N; i++) NoiseIcosahedron[i] = ivar_min;
   pixels_icosa_1(Npix, theta, phi, Nside, icosapix);
   for(i=0; i<Npix; i++) {
      NoiseIcosahedron[icosapix[i]] += 1./noisevar[i];
      if (OpSelect == 0) DataX.vector[icosapix[i]] += indata[i]/noisevar[i];
   }
   for(i=0; i<Icosahedron.N; i++) {
      NoiseIcosahedron[i] = 1./NoiseIcosahedron[i];
      DataX.vector[i] *= NoiseIcosahedron[i];
   }

   /* Perform coarse C^-1 operation */
   inverse_fftconv_icosa_1(&DataX, &DataCinvX, fftlen, ifd, Cl, lsplit, lbrutemax, &Icosahedron,
      OpSelect, tnum, FileName, NoiseIcosahedron, chebyshev_order);

   if (OpSelect == 0) {
      for(i=0; i<Icosahedron.N; i++) DataCinvX.vector[i] = NoiseIcosahedron[i] * DataCinvX.vector[i] - DataX.vector[i];
      for(i=0; i<Npix; i++) outdata[i] += (indata[i] + DataCinvX.vector[icosapix[i]])/noisevar[i];
   }

   /* Cleanup */
   free_lvector(icosapix, 0, Npix-1);
   free_dvector(NoiseIcosahedron, 0, Icosahedron.N-1);
   if (OpSelect == 4) {
      free_dmatrix(ifd[tnum].kernel_real, 0, fftlen-1, 0, fftlen-1);
      free_dmatrix(ifd[tnum].kernel_imag, 0, fftlen-1, 0, fftlen-1);
   }
   deallocate_icosahedral(&DataX);
   deallocate_icosahedral(&DataCinvX);
   deallocate_sphere_pixel(&Icosahedron);

#undef NFACTOR
}
#endif

/* read_preconditioner_data
 * *** READS A PRECONDITIONER DATAFILE ***
 *
 * Reads in a list of long integers from the datafile
 * The file format is, e.g.
 *
 * 3
 * 1 1 16
 * 4 2 16 256 96 10
 * 1 1 16
 *
 * The first line is the number of datasets.
 * Each following line contains the number of arguments, the
 * code for the preconditioner (see below),
 * and then the arguments themselves.
 *
 * The codes and number of arguments are:
 * Code  #arg   Routine
 * ====================================
 *   1      1   woodbury_preconditioner_1
 *   2      5   inverse_fftconv_icosa_2
 *   3      1   woodbury_preconditioner_2
 *
 * The file is read only on setup, thereafter it is not used.
 *
 * Arguments:
 *   FileName: file to read from
 *   setindex: which data set to get
 */
#ifndef THREAD_SAFE_SHT
long* read_preconditioner_data(char FileName[], int setindex) {

   FILE *fp;
   static int is_setup = 0;
   static long **data;
   static int *num_args;
   static int num_data_set;
   int j, i;

   /* Setup if not done already. */
   if (!is_setup) {
      fp = fopen(FileName, "r");

      /* Get number of datasets, allocate memory for the arguments
       * of the preconditioners (data) and the number of such arguments
       * (num_args).  We will use the j (dataset) index in unit-offset
       * convention, and i (which argument) in zero-offset.
       */
      fscanf(fp, "%d", &num_data_set);
      data = (long **)malloc((size_t) ((num_data_set+1)*sizeof(long*)));
      num_args = ivector(1, num_data_set);

      /* Now read each line ... */
      for(j=1;j<=num_data_set;j++) {
         fscanf(fp, "%d", num_args+j);
         data[j] = lvector(0, num_args[j]);
         for(i=0;i<=num_args[j];i++) fscanf(fp, "%ld", data[j]+i);
      }

      fclose(fp);
      is_setup = 1;

      printf("Loaded file: %s\n", FileName);
   }

   /* Return the arguments */
   return( data[setindex] );
}
#endif
