#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "sht.h"

/* This is a simple demo program that you can run:

$ gcc sht_demo.c sht.c -lm -o sht_demo.x
$ ./sht_demo.x

Comments are in the print statements.
*/

void errmessage(char message[]) {
   printf("%s\n", message);
   printf("Exiting.\n");
   exit(1);
}

int main(void) {
   SPHERE_MODES alm1;
   double *theta, *phi, *f, *g;
   long i,Npix, nsp, l, m, nx, ny;
   double a90, mu, sigma2, th, X, Y, Z, temp;
   double *ckern;
   clock_t t_start;

   printf("*** ILLUSTRATION WITH A FEW POINTS ***\n");

   printf("Longitude-fixing macro:\n");
   for(i=-13;i<14;i++) {
      printf("%6.2f ", (double)i);
   }
   printf("\n");
   for(i=-13;i<14;i++) {
      temp = (double)i;
      printf(" %6.3f", LONGITUDE_FIX(temp));
   }
   printf("\n");

   printf("Making spherical harmonic mode structures ...\n   ");
   allocate_sphere_modes(&alm1, 10);
   printf(" lmax=%2ld, Nmode=%3ld, coefs->%016lX, vector->%016lX\n\n",
      alm1.lmax, alm1.Nmode, (unsigned long)alm1.coefs, (unsigned long)alm1.vector);

   Npix = 19;
   printf("Making grid ... %ld points, spaced from 20N,60E to antipode @ heading 60deg N of E\n", Npix);
   printf("_i_ __psi__ _theta_ __phi__ (in deg)\n");
   theta = (double*)malloc((size_t)(Npix*sizeof(double)));
   if (!theta) errmessage("theta failed");
   phi = (double*)malloc((size_t)(Npix*sizeof(double)));
   if (!phi) errmessage("phi failed");
   for(i=0;i<Npix;i++) {
      printf("%3ld %7.3lf ", i, sphere_step_1(7.*M_PI/18., M_PI/3., M_PI*i/(Npix-1.), M_PI/3., theta+i, phi+i)*180./M_PI);
      printf("%7.3lf %7.3lf\n", theta[i]*180./M_PI, phi[i]*180./M_PI);
   }

   f = (double*)malloc((size_t)(2*Npix*sizeof(double)));
   if (!f) errmessage("allocation of f failed");
   g = f+Npix;
   for(i=0;i<Npix;i++) f[i]=0.;
   f[0]=1.;
   printf("Made function f: [");
   for(i=0;i<Npix;i++) printf(" %5.2f", f[i]);
   printf("]\n\n");

   for(nsp=32;nsp<=128;nsp*=2) {
      printf("Spherical harmonic conversion @ resolution %3ld\n", nsp);
      printf("   |   m= 0     m=1cos  m=1sin etc.\n");
      for(i=0;i<alm1.Nmode;i++) alm1.vector[i]=0.; /* need to initialize this */
      pixel_analysis_1(&alm1, NULL, f, NULL, alm1.lmax, theta, phi, NULL, NULL, Npix, NULL, NULL, 0, 0, nsp, 6);
      for(l=0;l<=alm1.lmax;l++) {
         printf("%2ld | %7.4lf  ", l, alm1.coefs[l][0]);
         for(m=1;m<=l;m++) printf(" %7.4lf %7.4lf", alm1.coefs[l][m], alm1.coefs[l][-m]);
         printf("\n");
      }
   }
   nsp/=2; /* set back to 160 */
   mu = cos(theta[0]);
   a90 = (12155*pow(mu,9)-25740*pow(mu,7)+18018*pow(mu,5)-4620*pow(mu,3)+315*mu)/128. / sqrt(4.*M_PI/19.);
   printf("Compare: direct a_{9,0} = %7.4lf [diff at last spacing=%11.4lE, nsp=%ld]\n", a90, a90-alm1.coefs[9][0], nsp);

   sigma2 = 0.09;
   printf("\nTransformation back to pixel grid, implementing Gaussian smoothing, var = %8.6f radian^2\n", sigma2);
   ckern = (double*)malloc((size_t)((alm1.lmax+1)*sizeof(double)));
   for(l=0;l<=alm1.lmax;l++) ckern[l] = exp(-0.5*l*(l+1)*sigma2);
   for(i=0;i<Npix;i++) g[i]=0.;
   pixel_synthesis_1(&alm1, NULL, g, NULL, alm1.lmax, theta, phi, NULL, Npix, ckern, NULL, 0, 0x1, nsp, 6);
   printf("output pixels: angle(rad), angle(deg), output amplitude, flat sky prediction\n");
   for(i=0;i<Npix;i++) {
      th = i/(Npix-1.)*M_PI;
      printf("%8.6f %8.4f %8.5f %8.5f\n", th, th*180./M_PI, g[i], exp(-th*th/2./sigma2)/(2*M_PI*sigma2));
   }

   printf("... and this is what happens if you keep calling pixel_analysis_1 without reinitializing\n");
   for(i=0;i<4;i++) {
      printf("a9,0=%9.6f\n", alm1.coefs[9][0]);
      pixel_analysis_1(&alm1, NULL, f, NULL, alm1.lmax, theta, phi, NULL, NULL, Npix, NULL, NULL, 0, 0, nsp, 6);
   }

   printf("Cleanup ... ");
   free((void*)theta);
   free((void*)phi);
   free((void*)f);
   free((void*)ckern);
   deallocate_sphere_modes(&alm1);
   printf("done.\n\n");


   printf("*** TIMING TESTS ***\n\n");
   t_start = clock();
   printf("clock started\n");
   nx=1500; ny=750;
   Npix = nx*ny;
   allocate_sphere_modes(&alm1, 500);
   printf("test with %ld pixels, lmax=%ld, nsp=%ld\n", Npix, alm1.lmax, nsp);
   theta = (double*)malloc((size_t)(Npix*sizeof(double)));
   if (!theta) errmessage("theta failed");
   phi = (double*)malloc((size_t)(Npix*sizeof(double)));
   if (!phi) errmessage("phi failed");
   f = (double*)malloc((size_t)(2*Npix*sizeof(double)));
   if (!f) errmessage("allocation of f failed");
   g = f+Npix;
   printf("setting up pixel grid. rectangular array, rotated 1e-4 radians around X axis\n");
   for(i=0;i<Npix;i++) {
      theta[i] = acos((i/nx+.5)/(double)ny-1);
      phi[i] = 2.*M_PI*(i%nx)/(double)nx;
      f[i] = sin(100*theta[i]+200*phi[i]); /* barber pole pattern */
      X = sin(theta[i])*cos(phi[i]);
      Y = sin(theta[i])*sin(phi[i]);
      Z = cos(theta[i]);
      temp = Y*cos(1e-4)+Z*sin(1e-4);
      Z = Z*cos(1e-4)-Y*sin(1e-4);
      Y = temp;
      phi[i] = LONGITUDE_FIX(atan2(Y,X));
      theta[i] = atan2(sqrt(X*X+Y*Y),Z);
      g[i] = 0.;
   }
   printf("constructed barber pole pattern:\n   f[375000 .. 375014] = [");
   for(i=0;i<15;i++) printf(" %8.5lf,", f[375000+i]);
   printf("]\n");
   ckern = (double*)malloc((size_t)((alm1.lmax+1)*sizeof(double)));
   for(l=0;l<=alm1.lmax;l++) ckern[l] = .01;
   ckern[400] = 1.01;
   printf("kernel constructed: ckern[...]=.01 except ckern[400]=1.01\n");
   printf("time: %8.3f s [setup arrays]\n", (double)(clock()-t_start)/CLOCKS_PER_SEC);
   nsp=4096;
   for(i=0;i<alm1.Nmode;i++) alm1.vector[i]=0.;
   pixel_analysis_1(&alm1, NULL, f, NULL, alm1.lmax, theta, phi, NULL, NULL, Npix, NULL, NULL, 0, 0, nsp, 6);
   for(m=-201;m<=-199;m++) {
      printf("spherical harmonic moments, l=[395..405] m=%ld:\n  [", m);
      for(l=395;l<=405;l++) printf(" %11.5lf", alm1.coefs[l][m]);
      printf("]\n");
   }
   for(m=199;m<=201;m++) {
      printf("spherical harmonic moments, l=[395..405] m=%ld:\n  [", m);
      for(l=395;l<=405;l++) printf(" %11.5lf", alm1.coefs[l][m]);
      printf("]\n");
   }
   printf("time: %8.3f s [ --> harmonic space]\n", (double)(clock()-t_start)/CLOCKS_PER_SEC);

   pixel_synthesis_1(&alm1, NULL, g, NULL, alm1.lmax, theta, phi, NULL, Npix, ckern, NULL, 0, 0x1, nsp, 6);
   printf("convolved pattern:\n   g[375000 .. 375014] = [");
   for(i=0;i<15;i++) printf(" %8.3lf,", g[375000+i]);
   printf("]\n");
   printf("time: %8.3f s [ --> real space]\n", (double)(clock()-t_start)/CLOCKS_PER_SEC);

   free((void*)theta);
   free((void*)phi);
   free((void*)f);
   free((void*)ckern);
   deallocate_sphere_modes(&alm1);
   printf("time: %8.3f s\n", (double)(clock()-t_start)/CLOCKS_PER_SEC);

   return(0);
}
