#!/usr/bin/env python

# This scripts takes 2 arguments, 1) Input power file 2) Output power file.
# It reads the power spectrum file with given z, k, power, error.
# It then creates a weighted smooth spline:
#   WITHOUT MASKING ANY POINTS,
#   But weighting each point with 1/error
# If --interp_log is passed as 3rd argument:
#   It MASKS P, e < 0. 
#   Interpolates (ln(1+z), lnk, lnP) with weights e/P
# Finally, it saves this smooth power in the same order to the output text file.

import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("InputPS", help="Input power spectrum file.")
    parser.add_argument("OutputPS", help="Output power spectrum file.")
    parser.add_argument("--interp_log", help="Interpolate ln(k), ln(P) instead.", action="store_true")
    args = parser.parse_args()
    
    # Read input power file.
    z, k, p, e = np.genfromtxt(args.InputPS, delimiter = ' ', skip_header = 2, unpack = True, usecols=(0,1,2,3))

    # Create 2D Spline object
    # From scipy manual:
    # Default s=len(weight) which should be a good value 
    # if 1/weight[i] is an estimate of the standard deviation of power[i].

    if args.interp_log:
        print("Smoothing ln(k), ln(P) and removing p,e<=0 points.")
        mask = np.logical_and(p > 0, e > 0)

        lnz = np.log(1+z[mask])
        lnk = np.log(k[mask])
        lnP = np.log(p[mask])
        lnE = e[mask]/p[mask]

        wsbispline = SmoothBivariateSpline(lnz, lnk, lnP, w=1./lnE, s=len(lnE))

        smwe_power = wsbispline(np.log(1+z), np.log(k), grid=False)
        smwe_power = np.exp(smwe_power)
    else:
        print("Smoothing k, P without masking any points.")
        wsbispline = SmoothBivariateSpline(z, k, p, w=1./e, s=len(e))

        smwe_power = wsbispline(z, k, grid=False)

    np.savetxt(args.OutputPS, smwe_power, header='', comments='')
    exit(0)
