#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include <memory>

#include "sht.hpp"

/* This is a simple demo program that you can run:

$ g++ sht_demo.cpp sht.c -lm -o sht_demo.xx
$ ./sht_demo.xx

*/

void errmessage(char message[]) {
   printf("%s\n", message);
   printf("Exiting.\n");
   exit(1);
}

const double RAD_TO_DEG = 180. / M_PI;

int main(void) {
   double *g;
   long i,Npix, nsp, l, m, nx, ny;
   double a90, mu, sigma2, th, X, Y, Z, temp;
   double *ckern;
   clock_t t_start;

   printf("Making spherical harmonic mode structures ...\n   ");
   nsp = 32;
   SHT mysht(10, 6, nsp);

   Npix = 19;
   printf("Making grid ... %ld points, spaced from 20N,60E to antipode @ heading 60deg N of E\n", Npix);
   printf("_i_ __psi__ _theta_ __phi__ (in deg)\n");
   auto theta = std::make_unique<double[]>(Npix),
        phi = std::make_unique<double[]>(Npix),
        f = std::make_unique<double[]>(2 * Npix);

   for (i = 0; i < Npix; i++) {
      printf("%3ld %7.3lf ", i, sphere_step_1(
         7. * M_PI / 18., M_PI / 3., M_PI * i / (Npix - 1.), M_PI / 3.,
         theta.get() + i, phi.get() + i) * RAD_TO_DEG);
      printf("%7.3lf %7.3lf\n", theta[i] * RAD_TO_DEG, phi[i] * RAD_TO_DEG);
   }

   // g = f.get() + Npix;
   f[0] = 1.0;
   printf("Made function f: [");
   for (i = 0; i < Npix; i++) printf(" %5.2f", f[i]);
   printf("]\n\n");

   printf("Spherical harmonic conversion @ resolution %3ld\n", nsp);
   printf("   |   m= 0     m=1cos  m=1sin etc.\n");

   mysht.zeroAlm();
   for (int i = 0; i < Npix; ++i)
      mysht.reverseInterpolate(theta[i], phi[i], f[i]);
   mysht.analysis();

   SPHERE_MODES alm1 = mysht.Alm;

   for(l=0;l<=alm1.lmax;l++) {
      printf("%2ld | %7.4lf  ", l, alm1.coefs[l][0]);
      for(m=1;m<=l;m++) printf(" %7.4lf %7.4lf", alm1.coefs[l][m], alm1.coefs[l][-m]);
      printf("\n");
   }

   return 0;
}
