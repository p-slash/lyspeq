import numpy as np
import argparse
import struct
import os

ABS_ERR = 1E-8
REL_ERR = 1E-5

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
    max_diff    = np.max(np.abs(a1 - a2))
    # max_rel_err = np.max(2*np.abs(a1 - a2)/np.abs(a1 + a2)) if max_diff<ABS_ERR else 0

    print("\tMaximum absolute error is {:.1e}.".format(max_diff))
    # print("\tMaximum relative error is {:.1e}.".format(max_rel_err))
    if max_diff > ABS_ERR: # or max_rel_err > REL_ERR:
        print(f"{bcolors.FAIL}{bcolors.BOLD}\tERROR: Error is above threshold!{bcolors.ENDC}")
        return 1

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("SourceDir", help="Source directory")
    args = parser.parse_args()
    ERR_CODE = 0

    print(f"{bcolors.HEADER}Comparing test results. Using absolute error of {ABS_ERR:.0e}.{bcolors.ENDC}")
    # print("Comparing test results. Using absolute error of {:.0e} and relative error of {:.0e}"\
    #    .format(ABS_ERR, REL_ERR))

    # TESTING SQ LOOKUP TABLES
    # Read signal tables
    print(f"{bcolors.BOLD}Comparing signal lookup tables...{bcolors.ENDC}")
    true_signal_table = readSQTable(args.SourceDir+"/tests/truth/signal_R70000_dv20.0.dat")
    comp_signal_table = readSQTable(args.SourceDir+"/tests/output/signal_R70000_dv20.0.dat")
    ERR_CODE += testMaxDiffArrays(true_signal_table, comp_signal_table)
    del true_signal_table, comp_signal_table

    print(f"{bcolors.BOLD}Comparing derivative lookup tables...{bcolors.ENDC}")
    TRUTH_DIR = os.path.join(args.SourceDir, "tests/truth")
    deriv_flist = [os.path.join(TRUTH_DIR, ff)  for ff in os.listdir(TRUTH_DIR) \
                    if ff.startswith("deriv_R70000_dv20.0")]
    for drv_true_file in deriv_flist:
        drv_comp_file = drv_true_file.replace("truth", "output")
        print("\tFile:", drv_comp_file)
        true_deriv_table = readSQTable(drv_true_file)
        comp_deriv_table = readSQTable(drv_comp_file)
        ERR_CODE += testMaxDiffArrays(true_deriv_table, comp_deriv_table)
    
    del true_deriv_table, comp_deriv_table

    # TESTING INTERPOLATED MATRICES
    # Read and compare interpolated fiducial signal matrix values
    print(f"{bcolors.BOLD}Comparing fiducial signal matrices...{bcolors.ENDC}")
    true_fid_signal = np.genfromtxt(args.SourceDir+"/tests/truth/signal_matrix.txt", skip_header=1)
    comp_fid_signal = np.genfromtxt(args.SourceDir+"/tests/output/signal_matrix.txt", skip_header=1)
    ERR_CODE += testMaxDiffArrays(true_fid_signal, comp_fid_signal)
    del true_fid_signal, comp_fid_signal

    # Read and compare interpolated derivative matrix values
    print(f"{bcolors.BOLD}Comparing derivative matrices...{bcolors.ENDC}")
    true_q0 = np.genfromtxt(args.SourceDir+"/tests/truth/q0_matrix.txt", skip_header=1)
    comp_q0 = np.genfromtxt(args.SourceDir+"/tests/output/q0_matrix.txt", skip_header=1)
    ERR_CODE += testMaxDiffArrays(true_q0, comp_q0)
    del true_q0, comp_q0

    # TESTING THE FINAL RESULTS
    print(f"{bcolors.BOLD}Comparing QMLE results...{bcolors.ENDC}")
    true_Pfid, true_ThetaP, true_ErrorP = \
    readQMLEResults(args.SourceDir+"/tests/truth/test_it1_quadratic_power_estimate_detailed.dat")
    comp_Pfid, comp_ThetaP, comp_ErrorP = \
    readQMLEResults(args.SourceDir+"/tests/output/test_it1_quadratic_power_estimate_detailed.dat")

    print("1. Fiducial power:")
    ERR_CODE += testMaxDiffArrays(true_Pfid, comp_Pfid)
    print("2. Theta estimates:")
    ERR_CODE += testMaxDiffArrays(true_ThetaP, comp_ThetaP)
    print("3. Error estimates:")
    ERR_CODE += testMaxDiffArrays(true_ErrorP, comp_ErrorP)

    print("4. Comparing Fisher matrices...")
    true_fisher = np.genfromtxt(args.SourceDir+"/tests/truth/test_it1_fisher_matrix.dat", skip_header=1)
    comp_fisher = np.genfromtxt(args.SourceDir+"/tests/output/test_it1_fisher_matrix.dat", skip_header=1)
    ERR_CODE += testMaxDiffArrays(true_fisher, comp_fisher)

    if ERR_CODE == 0:
        print(f"{bcolors.OKBLUE}{bcolors.BOLD}Everything works OK!{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}{bcolors.BOLD}There are {ERR_CODE:d} errors!{bcolors.ENDC}")
        
    exit(ERR_CODE)




