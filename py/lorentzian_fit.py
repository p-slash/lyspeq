#!/usr/bin/python

import sys

import numpy as np
from scipy.optimize import curve_fit

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
            + (B + beta * lnk) * lnz;

    return np.exp(lnkP_pi);

def pd13_lorentzian(X, A, n, alpha, B, beta, lmd):
    k, z = X
    return pd13form_fitting_function(X, A, n, alpha, B, beta)  / np.power((1. + k/2.), lmd)

input_ps = sys.argv[1]
output_ps= sys.argv[2]

f = open(input_ps, 'r')
string_sizes = f.readline()
size = [int(n) for n in string_sizes.split()]
f.close()

NzBins = size[0]
NkBins = size[1]

z, k, p, e = np.genfromtxt(input_ps, delimiter = ' ', skip_header = 2, unpack = True)

p = p * k / np.pi
e = e * k / np.pi

theoretical_ps = np.zeros(NzBins * NkBins)

# Fit using the same pd13 form, get new params
pd13_0 = 0.0662, -2.685, -0.223, 3.591, -0.177, 359.8

for nz in range(NzBins):
    ind_1 = nz * NkBins
    ind_2 = (nz + 1) * NkBins

    z1 = z[ind_1:ind_2]
    k1 = k[ind_1:ind_2]
    p1 = p[ind_1:ind_2]
    e1 = e[ind_1:ind_2]

    mask = p1 < 0
    
    z1_m = z1[~mask]
    k1_m = k1[~mask]
    p1_m = p1[~mask]
    e1_m = e1[~mask]

    pnew, pcov = curve_fit(pd13_lorentzian, (k1_m,z1_m), p1_m, pd13_0, sigma=e1_m)

    theoretical_ps[ind_1:ind_2] = pd13_lorentzian((k1,z1), *pnew)

np.savetxt(output_ps, theoretical_ps)
    
exit(0)
    
    # r                = p1 - theoretical_ps
    # chisq            = np.sum((r/error)**2)
    # df               = len(power_sp) - 6

    # fit_param_text = """A        = %.3e
    # n        = %.3e
    # alpha    = %.3e
    # B        = %.3e
    # beta     = %.3e
    # lambda   = %.3e""" % (pnew[0], pnew[1], pnew[2], pnew[3], pnew[4], pnew[5])
    # print(fit_param_text)
    # print("chisq = ", chisq, "doff = ",df, "chisq/dof = ", chisq/df)


