/* SPHERE_MODES structure, allocation and de-allocation function */
typedef struct {
   long lmax;
   long Nmode;
   double **coefs;
   double *vector;
} SPHERE_MODES;

void allocate_sphere_modes(SPHERE_MODES *A, long lmax);
void deallocate_sphere_modes(SPHERE_MODES *A);

/* 2 main functions:
 * pixel_synthesis_1, spherical harmonic space to pixel space
 * pixel_analysis_1, pixel space to spherical harmonic space
 *
 * Both functions *add* the result to the destination structure.
 *
 * Two components can be specified for spin=1 or spin=2 transforms
 * spin=1 -> vector & axial harmonics, X & Y components
 * spin=2 -> E & B harmonics, Q & U components
 * For spin=0 (scalar), n_LM and component_2 are not used and can be NULL
 *
 * p_LM and n_LM should be SPHERE_MODES structures with the specified lmax
 * component_1 and component_2 are the real space maps and are arrays of size length
 * the pixel positions are at the given theta, phi (standard spherical coordinates) and rotation angle psi
 * (0 = X to East, Y to North; pi/2 = X to North, Y to West)
 *
 * convolution options:
 * convolve_kernel_p and convolve_kernel_n are arrays of size lmax+1 (multiplies in C_l space) [if convolve_flag & 0x1]
 * area: array of size length, multiply by pixel area (in real->harmonic space only) [if convolve_flag & 0x2]
 * all of these are turned off if convolve_flag = 0
 *
 * resolution option:
 * nspace = number of grid points around the sphere. The grid spacing delta theta = delta phi = 2*pi/nspace
 * pm_order = particle-mesh interpolation polynomial order (max=10)
 */

void pixel_synthesis_1(SPHERE_MODES *p_LM, SPHERE_MODES *n_LM, double *component_1, double *component_2,
   long lmax, double *theta, double *phi, double *psi, long length, double *convolve_kernel_p,
   double *convolve_kernel_n, int spin, unsigned short int convolve_flag, long nspace, long pm_order);

void pixel_analysis_1(SPHERE_MODES *p_LM, SPHERE_MODES *n_LM, double *component_1, double *component_2,
   long lmax, double *theta, double *phi, double *psi, double *area, long length, double *convolve_kernel_p,
   double *convolve_kernel_n, int spin, unsigned short int convolve_flag, long nspace, long pm_order);

/* All the functions require phi to be in the range [0,2pi). Here's a helper macro that can put
 * a longitude in the right range.
 */
#define LONGITUDE_FIX(x) ((x)-2*M_PI*floor((x)/(2*M_PI)))

/* This is a routine to move a point from P0(theta0,phi0) --> P(theta,phi) by moving an angle lstep in the
 * direction given by heading. The heading is in the convention 0 (East), pi/2 (North), etc. (so pi/2 minus position angle).
 * The return value is the rotation psi of the coordinate system by transport along great circle arc,
 * in the sense that positive psi means "East" at P is "psi radians North of East" at P0.
 */
double sphere_step_1(double theta0, double phi0, double lstep, double heading, double *theta, double *phi);

