import numpy as np
import argparse
import struct
import os

NUM_PREC = 1E-8

def readSQTable(fname):
    file  = open(fname, mode='rb')
    header_fmt  = 'iiddiddd'
    header_size = struct.calcsize(header_fmt)

    d = file.read(header_size)

    nv, nz, Lv, Lz, R, dv, k1, k2 = struct.unpack(header_fmt, d)

    size = nv if nz==0 else nv*nz

    array_fmt  = 'd' * size
    array_size = struct.calcsize(array_fmt)

    d = file.read(array_size)

    return np.array(struct.unpack(array_fmt, d))

def readQMLEResults(fname):
    with open(fname) as f:
        lines = (line for line in f if not line.startswith('#'))
        return np.loadtxt(lines, skiprows=1, usecols=(4, 5, 7), unpack=True)

def testMaxDiffArrays(a1, a2):
    max_diff = np.max(np.abs(a1 - a2))
    print("Maximum difference is {:.1e}.".format(max_diff))
    if max_diff > NUM_PREC:
        print("ERROR: Greater than {:.1e}!".format(NUM_PREC))
        return 1

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("SourceDir", help="Source directory")
    args = parser.parse_args()
    ERR_CODE = 0

    # TESTING SQ LOOKUP TABLES
    # Read signal tables
    print("Comparing signal lookup tables...")
    true_signal_table = readSQTable(args.SourceDir+"/tests/truth/signal_R70000.dat")
    comp_signal_table = readSQTable(args.SourceDir+"/tests/input/signal_R70000.dat")
    ERR_CODE += testMaxDiffArrays(true_signal_table, comp_signal_table)
    del true_signal_table, comp_signal_table

    print("Comparing derivative lookup tables...")
    TRUTH_DIR = os.path.join(args.SourceDir, "tests/truth")
    deriv_flist = [os.path.join(TRUTH_DIR, ff)  for ff in os.listdir(TRUTH_DIR) \
                    if ff.startswith("deriv_R70000")]
    for drv_true_file in deriv_flist:
        drv_comp_file = drv_true_file.replace("truth", "input")
        print("File:", drv_comp_file)
        true_deriv_table = readSQTable(drv_true_file)
        comp_deriv_table = readSQTable(drv_comp_file)
        ERR_CODE += testMaxDiffArrays(true_deriv_table, comp_deriv_table)
    
    del true_deriv_table, comp_deriv_table

    # TESTING INTERPOLATED MATRICES
    # Read and compare interpolated fiducial signal matrix values
    print("Comparing fiducial signal matrices...")
    true_fid_signal = np.genfromtxt(args.SourceDir+"/tests/truth/signal_matrix.txt", skip_header=1)
    comp_fid_signal = np.genfromtxt(args.SourceDir+"/tests/output/signal_matrix.txt", skip_header=1)
    ERR_CODE += testMaxDiffArrays(true_fid_signal, comp_fid_signal)
    del true_fid_signal, comp_fid_signal

    # Read and compare interpolated derivative matrix values
    print("Comparing derivative matrices...")
    true_q0 = np.genfromtxt(args.SourceDir+"/tests/truth/q0_matrix.txt", skip_header=1)
    comp_q0 = np.genfromtxt(args.SourceDir+"/tests/output/q0_matrix.txt", skip_header=1)
    ERR_CODE += testMaxDiffArrays(true_q0, comp_q0)
    del true_q0, comp_q0

    # TESTING THE FINAL RESULTS
    print("Comparing QMLE results...")
    true_Pfid, true_ThetaP, true_ErrorP = \
    readQMLEResults(args.SourceDir+"/tests/truth/test_it1_quadratic_power_estimate_detailed.dat")
    comp_Pfid, comp_ThetaP, comp_ErrorP = \
    readQMLEResults(args.SourceDir+"/tests/output/test_it1_quadratic_power_estimate_detailed.dat")

    print("Fiducial power:")
    ERR_CODE += testMaxDiffArrays(true_Pfid, comp_Pfid)
    print("Theta estimates:")
    ERR_CODE += testMaxDiffArrays(true_ThetaP, comp_ThetaP)
    print("Error estimates:")
    ERR_CODE += testMaxDiffArrays(true_ErrorP, comp_ErrorP)

    print("Comparing Fisher matrices...")
    true_fisher = np.genfromtxt(args.SourceDir+"/tests/truth/test_it1_fisher_matrix.dat", skip_header=1)
    comp_fisher = np.genfromtxt(args.SourceDir+"/tests/output/test_it1_fisher_matrix.dat", skip_header=1)
    ERR_CODE += testMaxDiffArrays(true_fisher, comp_fisher)

    exit(ERR_CODE)




