#!/usr/bin/python

import sys

import numpy as np
from scipy.optimize import curve_fit

# Fit using the same fiducial parameters
fiducial_params = 0.0662, -2.685, -0.223, 3.591, -0.177, 360.

def pd13form_fitting_function(X, A, n, alpha, B, beta):
    k, z = X

    k_0 = 0.009; # s/km
    z_0 = 3.0;

    lnk = np.log(k / k_0)
    lnz = np.log((1. + z) / (1. + z_0))
    lnkP_pi = 0;

    lnkP_pi = np.log(A) \
            + (3. + n) * lnk \
            + alpha * lnk * lnk \
            + (B + beta * lnk) * lnz

    return np.exp(lnkP_pi) * np.pi / k

def pd13_lorentzian(X, A, n, alpha, B, beta, lmd):
    k, z = X
    return pd13form_fitting_function(X, A, n, alpha, B, beta)  / (1. + lmd * k**2)

def pd13_lorentzian_noz(k, A, n, alpha, lmd):
    X = (k, 3.)
    return pd13_lorentzian(X, A, n, alpha, 0, 0, lmd)

input_ps = sys.argv[1]
output_ps= sys.argv[2]

if len(sys.argv) == 9:
    fiducial_params = float(sys.argv[3:])

z, k, p, e = np.genfromtxt(input_ps, delimiter = ' ', skip_header = 2, unpack = True)

theoretical_ps = np.zeros(len(z))

# Global Fit
mask     = np.greater(p, 0.)
z_masked = z[mask]
k_masked = k[mask]
p_masked = p[mask]
e_masked = e[mask]

try:
    pnew, pcov = curve_fit(pd13_lorentzian, (k_masked, z_masked), p_masked, fiducial_params, sigma=e_masked)
except ValueError:
    print("ValueError: Either ydata or xdata contain NaNs")
    exit(1)
except RuntimeError:
    print("RuntimeError: The least-squares minimization fails")
    exit(1)
except OptimizeWarning:
    print("OptimizeWarning: Covariance of the parameters can not be estimated")    
    exit(1)

theoretical_ps = pd13_lorentzian((k, z), *pnew)

r              = p_masked - theoretical_ps[mask]
chisq          = np.sum((r/e_masked)**2)
df             = len(p_masked) - 6

fit_param_text = """A        = %.3e
n        = %e
alpha    = %e
B        = %e
beta     = %e
lambda   = %e""" % (pnew[0], pnew[1], pnew[2], pnew[3], pnew[4], pnew[5])
print(fit_param_text)
print("chisq = ", chisq, "doff = ",df, "chisq/dof = ", chisq/df)

params_header = "%e %e %e %e %e %e" % (pnew[0], pnew[1], pnew[2], pnew[3], pnew[4], pnew[5])
np.savetxt(output_ps, theoretical_ps, header=params_header)
    
exit(0)

# for nz in range(NzBins):
#     ind_1 = nz * NkBins
#     ind_2 = (nz + 1) * NkBins

#     k1 = k[ind_1:ind_2]
#     p1 = p[ind_1:ind_2]
#     e1 = e[ind_1:ind_2]

#     mask = np.greater(p1, 0)
    
#     k1_m = k1[mask]
#     p1_m = p1[mask]
#     e1_m = e1[mask]

#     pnew, pcov = curve_fit(pd13_lorentzian_noz, k1_m, p1_m, pd13_0, sigma=e1_m)

#     theoretical_ps[ind_1:ind_2] = pd13_lorentzian_noz(k1, *pnew)


    
    


