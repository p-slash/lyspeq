#!/usr/bin/env python

import numpy as np
import argparse
import struct

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("SourceDir", help="Source directory")
    args = parser.parse_args()
    ERR_CODE = 0
    NUM_PREC = 1E-8

    # Read and compare interpolated fiducial signal matrix values
    print("Comparing fiducial signal matrices...")
    true_fid_signal = np.genfromtxt(args.SourceDir+"/tests/truth/signal_matrix.txt", skip_header=1)
    comp_fid_signal = np.genfromtxt(args.SourceDir+"/tests/output/signal_matrix.txt", skip_header=1)

    max_diff = np.max(np.abs(true_fid_signal - comp_fid_signal))
    print("Maximum difference is {:.1e}.".format(max_diff))
    if max_diff > NUM_PREC:
        print("ERROR: Greater than {:.1e}!".format(NUM_PREC))
        ERR_CODE=1

    # Read and compare interpolated derivative matrix values
    print("Comparing derivative matrices...")

    true_q0 = np.genfromtxt(args.SourceDir+"/tests/truth/q0_matrix.txt", skip_header=1)
    comp_q0 = np.genfromtxt(args.SourceDir+"/tests/output/q0_matrix.txt", skip_header=1)

    max_diff = np.max(np.abs(true_q0 - comp_q0))
    print("Maximum difference is {:.1e}.".format(max_diff))
    if max_diff > NUM_PREC:
        print("ERROR: Greater than {:.1e}!".format(NUM_PREC))
        ERR_CODE=1

    exit(ERR_CODE)