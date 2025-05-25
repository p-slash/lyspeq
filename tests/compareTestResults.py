import numpy as np
import argparse
import struct

import fitsio

ABS_ERR = 1E-8
REL_ERR = 1E-5


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def readSQTable(fname):
    file = open(fname, mode='rb')
    header_fmt = 'iidddiddd'
    header_size = struct.calcsize(header_fmt)

    d = file.read(header_size)

    nv, nz, Lv, z1, Lz, R, dv, k1, k2 = struct.unpack(header_fmt, d)

    size = nv if nz == 0 else nv * nz

    array_fmt = 'd' * size
    array_size = struct.calcsize(array_fmt)

    d = file.read(array_size)

    return np.array(struct.unpack(array_fmt, d))


def readQMLEResults(fname):
    with open(fname) as f:
        lines = (line for line in f if not line.startswith('#'))
        return np.loadtxt(lines, skiprows=1, usecols=(4, 5, 7), unpack=True)


def testMaxDiffArrays(a1, a2):
    diff = np.abs(a1 - a2)
    mag = np.maximum(np.abs(a1), np.abs(a2))
    budget = ABS_ERR + REL_ERR * mag
    offset = diff - budget
    p = np.all(offset < 0)
    max_diff = np.max(diff)
    rel_diff = np.max(diff / mag)

    print(
        f"\tMaximum absolute error is {max_diff:.1e}.\n"
        f"\tMaximum relative error is {rel_diff:.1e}.")
    if not p:
        print(f"{FAIL}{BOLD}\tERROR: Error is above threshold!{ENDC}")
        return 1

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("SourceDir", help="Source directory")
    args = parser.parse_args()
    ERR_CODE = 0

    print(
        f"{HEADER}Comparing test results. "
        f"Using absolute error of {ABS_ERR:.0e}.{ENDC}")

    # TESTING SQ LOOKUP TABLES
    # Read signal tables
    print(f"{BOLD}Comparing signal lookup tables...{ENDC}")
    true_signal_table = fitsio.read(
        args.SourceDir + "/tests/truth/signal_R70000_dv20.0.fits",
        ext="SIGNAL")
    comp_signal_table = fitsio.read(
        args.SourceDir + "/tests/output/signal_R70000_dv20.0.fits",
        ext="SIGNAL")
    ERR_CODE += testMaxDiffArrays(true_signal_table, comp_signal_table)
    del true_signal_table, comp_signal_table

    print(f"{BOLD}Comparing derivative lookup tables...{ENDC}")
    true_deriv_table = fitsio.read(
        args.SourceDir + "/tests/truth/signal_R70000_dv20.0.fits",
        ext="DERIVATIVE")
    comp_deriv_table = fitsio.read(
        args.SourceDir + "/tests/output/signal_R70000_dv20.0.fits",
        ext="DERIVATIVE")

    ERR_CODE += testMaxDiffArrays(true_deriv_table, comp_deriv_table)

    del true_deriv_table, comp_deriv_table

    # TESTING THE FINAL RESULTS
    print(f"{BOLD}Comparing QMLE results...{ENDC}")
    true_Pfid, true_ThetaP, true_ErrorP = readQMLEResults(
        f"{args.SourceDir}/tests/truth/test_it1_quadratic_power_estimate_detailed.dat")
    comp_res = fitsio.read(
        f"{args.SourceDir}/tests/output/test_detailed_results.fits",
        ext="P1D")
    comp_Pfid, comp_ThetaP, comp_ErrorP = comp_res['PINPUT'], comp_res['ThetaP'], comp_res['E_PK']

    print("1. Fiducial power:")
    ERR_CODE += testMaxDiffArrays(true_Pfid, comp_Pfid)
    print("2. Theta estimates:")
    ERR_CODE += testMaxDiffArrays(true_ThetaP, comp_ThetaP)
    print("3. Error estimates:")
    ERR_CODE += testMaxDiffArrays(true_ErrorP, comp_ErrorP)

    print("4. Comparing Fisher matrices...")
    true_fisher = fitsio.read(
        f"{args.SourceDir}/tests/truth/test_it1_matrices.fits",
        ext="FISHER")
    comp_fisher = fitsio.read(
        f"{args.SourceDir}/tests/output/test_detailed_results.fits",
        ext="FISHER")
    ERR_CODE += testMaxDiffArrays(true_fisher, comp_fisher)

    if ERR_CODE == 0:
        print(
            f"{OKBLUE}{BOLD}Everything works OK!{ENDC}")
    else:
        print(
            f"{FAIL}{BOLD}There are {ERR_CODE:d} errors!{ENDC}")

    exit(ERR_CODE)
