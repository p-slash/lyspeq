# TODO:
# Add jacobian 

import sys
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)

K_0 = 0.009
Z_0 = 3.0

def pd13_lorentzian_fitting_function_k(k, A, n, alpha, lmd):
    lnk = np.log(k / K_0)

    lnkP_pi = np.log(A) \
            + (3. + n) * lnk \
            + alpha * lnk * lnk

    return np.exp(lnkP_pi) * np.pi / k / (1. + lmd * k**2)

def pd13_lorentzian_fitting_function_z(X, A, n, alpha, B, beta, lmd):
    k, z = X

    lnk = np.log(k / K_0)
    lnz = np.log((1. + z) / (1. + Z_0))
    
    p_z_mult = np.exp((B + beta * lnk) * lnz)

    return pd13_fitting_function_k(k, A, n, alpha, lmd) * p_z_mult

# def jacobian_k(k, A, n, alpha):
#     jac_k_res = np.zeros(3)

input_ps  = sys.argv[1]
output_ps = sys.argv[2]

if len(sys.argv) == 9:
    Z_EVO_ON        = True
    NUMBER_OF_PARAMS= 6
    fiducial_params = np.array(sys.argv[3:], dtype=float)
elif len(sys.argv) == 7:
    Z_EVO_ON        = False
    NUMBER_OF_PARAMS= 4
    fiducial_params = np.array(sys.argv[3:], dtype=float)
else:
    print("Not enough arguments. Pass fiducial cosmology parameters.")
    exit(1)

z, k, p, e = np.genfromtxt(input_ps, delimiter = ' ', skip_header = 2, unpack = True)

theoretical_ps = np.zeros(len(z))

# Global Fit
mask     = np.logical_and(np.greater(p, 0.), np.greater(e, 0.))
z_masked = z[mask]
k_masked = k[mask]
p_masked = p[mask]
e_masked = e[mask]

if Z_EVO_ON:
    fit_function = pd13_lorentzian_fitting_function_z
    X_masked     = (k_masked, z_masked)
    X            = (k, z)
else:
    fit_function = pd13_lorentzian_fitting_function_k
    X_masked     = k_masked
    X            = k
        
try:
    lb = [0, np.full(NUMBER_OF_PARAMS-2, -np.inf), 0]
    pnew, pcov = curve_fit(fit_function, X_masked, p_masked, fiducial_params, \
                            sigma=e_masked, bounds=(lb, np.inf))
except ValueError:
    print("ValueError: Either ydata or xdata contain NaNs.")
    exit(1)
except RuntimeError:
    print("RuntimeError: The least-squares minimization fails.")
    exit(1)
except OptimizeWarning:
    print("OptimizeWarning: Covariance of the parameters can not be estimated. Using fiducial parameters instead.")    
    pnew = fiducial_params

theoretical_ps = fit_function(X, *pnew)

r              = p_masked - theoretical_ps[mask]
chisq          = np.sum((r/e_masked)**2)
df             = len(p_masked) - NUMBER_OF_PARAMS

pnew_toprint = np.zeros(6)
if Z_EVO_ON:
    pnew_toprint = pnew
else:
    pnew_toprint[0:3] = pnew[0:3]
    pnew_toprint[5]   = pnew[3]

fit_param_text = """A        = %.3e
n        = %e
alpha    = %e
B        = %e
beta     = %e
lambda   = %e""" % (pnew_toprint[0], pnew_toprint[1], pnew_toprint[2], \
                    pnew_toprint[3], pnew_toprint[4], pnew_toprint[5])
print(fit_param_text)
print("chisq = ", chisq, "doff = ",df, "chisq/dof = ", chisq/df)

params_header = "%e %e %e %e %e %e" % ( pnew_toprint[0], pnew_toprint[1], pnew_toprint[2], \
                                        pnew_toprint[3], pnew_toprint[4], pnew_toprint[5])
np.savetxt(output_ps, theoretical_ps, header=params_header, comments='')
    
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


    
# def pd13form_fitting_function(X, A, n, alpha, B, beta):
#     k, z = X

#     lnk = np.log(k / k_0)
#     lnz = np.log((1. + z) / (1. + z_0))
#     lnkP_pi = 0;

#     lnkP_pi = np.log(A) \
#             + (3. + n) * lnk \
#             + alpha * lnk * lnk \
#             + (B + beta * lnk) * lnz

#     return np.exp(lnkP_pi) * np.pi / k   


