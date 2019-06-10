#!/usr/bin/env python

import sys
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)

K_0 = 0.009
Z_0 = 3.0

# Define PD13 fitting function with Lorentzian smooting
def pd13_lorentzian_fitting_function_k(k, A, n, alpha, lmd):
    q0 = k / K_0 + 1e-10
    
    return (A * np.pi / K_0) * np.power(q0, 2. + n + alpha * np.log(q0)) / (1. + lmd * k**2)

def pd13_lorentzian_fitting_function_z(X, A, n, alpha, B, beta, lmd):
    k, z = X
    
    q0 = k / K_0 + 1e-10
    x0 = (1. + z) / (1. + Z_0)

    p_z_mult = np.power(x0, B + beta * np.log(q0))

    return pd13_lorentzian_fitting_function_k(k, A, n, alpha, lmd) * p_z_mult

# Define their Jacobians
def jacobian_k(k, A, n, alpha, lmd):
    lnk = np.log(k / K_0)
    p_k = pd13_lorentzian_fitting_function_k(k, A, n, alpha, lmd)

    return np.column_stack((p_k / A, p_k * lnk, p_k * lnk * lnk, -p_k * k**2 / (1. + lmd * k**2)))

def jacobian_z(X, A, n, alpha, B, beta, lmd):
    k, z = X

    lnk = np.log(k / K_0)
    lnz = np.log((1. + z) / (1. + Z_0))

    p_z = pd13_lorentzian_fitting_function_z(X, A, n, alpha, B, beta, lmd)

    return np.column_stack((p_z / A  , p_z * lnk      ,  p_z * lnk * lnk, \
                            p_z * lnz, p_z * lnk * lnz, -p_z * k**2 / (1. + lmd * k**2)))

if __name__ == '__main__':
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

    z, k, p, e = np.genfromtxt(input_ps, delimiter = ' ', skip_header = 2, unpack = True, usecols=(0,1,2,3))

    theoretical_ps = np.zeros(len(z))

    # Global Fit
    mask     = np.logical_and(p > 0, e > 0)
    z_masked = z[mask]
    k_masked = k[mask]
    p_masked = p[mask]
    e_masked = e[mask]

    if Z_EVO_ON:
        fit_function = pd13_lorentzian_fitting_function_z
        jac_function = jacobian_z
        X_masked     = (k_masked, z_masked)
        X            = (k, z)
    else:
        fit_function = pd13_lorentzian_fitting_function_k
        jac_function = jacobian_k
        X_masked     = k_masked
        X            = k

    try:
        lb     = np.full(NUMBER_OF_PARAMS, -np.inf)
        lb[0]  = 0
        lb[-1] = 0

        pnew, pcov = curve_fit(fit_function, X_masked, p_masked, fiducial_params, \
                                sigma=e_masked, absolute_sigma=True, bounds=(lb, np.inf), \
                                method='trf', jac=jac_function)
    except ValueError:
        raise
        exit(1)
    except RuntimeError:
        raise
        exit(1)
    except OptimizeWarning:
        raise
        print("Using fiducial parameters instead.")
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
n        = %.3e
alpha    = %.3e
B        = %.3e
beta     = %.3e
lambda   = %.3e""" % (pnew_toprint[0], pnew_toprint[1], pnew_toprint[2], \
                        pnew_toprint[3], pnew_toprint[4], pnew_toprint[5])
    print(fit_param_text)
    print("chisq = %.2f"%chisq, "dof = ", df, "chisq/dof = %.2f"%(chisq/df))

    params_header = "%e %e %e %e %e %e" % ( pnew_toprint[0], pnew_toprint[1], pnew_toprint[2], \
                                            pnew_toprint[3], pnew_toprint[4], pnew_toprint[5])
    np.savetxt(output_ps, theoretical_ps, header=params_header, comments='')

    exit(0)

