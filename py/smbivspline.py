#!/usr/bin/env python

# This scripts takes 2 arguments, 1) Input power file 2) Output power file.
# It reads the power spectrum file with given z, k, power, error.
# It then creates a weighted smooth spline:
#   WITHOUT MASKING ANY POINTS,
#   But weighting each point with 1/error
# Finally, it saves this smooth power in the same order to the output text file.

from sys import argv as sys_argv

import numpy as np
from scipy.interpolate import SmoothBivariateSpline

if __name__ == '__main__':
    input_ps  = sys_argv[1]
    output_ps = sys_argv[2]

    # Read input power file.
    z, k, p, e = np.genfromtxt(input_ps, delimiter = ' ', skip_header = 2, unpack = True, usecols=(0,1,2,3))

    # Create 2D Spline object
    # From scipy manual:
    # Default s=len(weight) which should be a good value 
    # if 1/weight[i] is an estimate of the standard deviation of power[i].
    wsbispline = SmoothBivariateSpline(z, k, p, w=1./e, s=len(e))

    smwe_power = wsbispline(z, k, grid=False)

    np.savetxt(output_ps, smwe_power, header='', comments='')
    exit(0)
