#include "core/sq_table.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm> // std::copy
#include <stdexcept>

#include "core/fiducial_cosmology.hpp"
#include "core/mpi_manager.hpp"
#include "core/omp_manager.hpp"
#include "mathtools/fourier_integrator.hpp"
#include "mathtools/real_field.hpp"
#include "io/io_helper_functions.hpp"
#include "io/sq_lookup_table_file.hpp"
#include "io/logger.hpp"

std::unique_ptr<SQLookupTable> process::sq_private_table;

std::unique_ptr<double[]>
LINEAR_V_ARRAY, LINEAR_Z_ARRAY, signal_array;

void deallocateSignalAndDerivArrays()
{
    signal_array.reset();
}

void deallocateVAndZArrays()
{
    LINEAR_V_ARRAY.reset();

    if (!specifics::TURN_OFF_SFID)
        LINEAR_Z_ARRAY.reset();
}


// Internal Functions and Variables
namespace sqhelper
{
    double getLinearSpacing(double length, int N)
    {
        double r = (N==1) ? 0 : (length / (N - 1.));
        return r;
    }

    // Allocates a double array with size N
    // Must deallocate after!!
    std::unique_ptr<double[]> allocLinearlySpacedArray(double x0, double lengthx, int N)
    {
        auto resulting_array = std::make_unique<double[]>(N);
        double dx = getLinearSpacing(lengthx, N);

        for (int j = 0; j < N; ++j)
            resulting_array[j] = x0 + dx * j;

        return resulting_array;
    }

    inline
    bool isClose(double a, double b, double relerr=1e-5, double abserr=1e-8)
    {
        double mag = std::max(fabs(a),fabs(b));
        return fabs(a-b) < (abserr + relerr * mag);
    }

    void _checkHeaderConsistency(
            const SQ_IO_Header &hdr,
            int nvpts, int nzpts, double vlen, double z1, double zlen,
            int Rthis, double dvthis
    ) {
        std::vector<std::string> inconsistencies;
        if (hdr.v_length < specifics::MAX_FOREST_LENGTH_V)
            inconsistencies.push_back("v_length ");
        if (hdr.z1 > bins::Z_LOWER_EDGE)
            inconsistencies.push_back("z1 ");
        if ((hdr.z1 + hdr.z_length) < bins::Z_UPPER_EDGE)
            inconsistencies.push_back("z_length ");
        if (hdr.spectrograph_resolution != Rthis)
            inconsistencies.push_back("spectrograph_resolution ");
        if (!isClose(hdr.pixel_width, dvthis))
            inconsistencies.push_back("pixel_width ");

        if (!isClose(hdr.k1, bins::KBAND_EDGES[0]))
            inconsistencies.push_back("k1 ");
        if (!isClose(hdr.dklin, bins::DKLIN))
            inconsistencies.push_back("dklin ");
        if (!isClose(hdr.dklog, bins::DKLOG))
            inconsistencies.push_back("dklog ");

        if (hdr.nklin != bins::NKLIN)
            inconsistencies.push_back("nklin ");
        if (hdr.nklog != bins::NKLOG)
            inconsistencies.push_back("nklog ");

        if (!inconsistencies.empty()) {
            int total_chars = 1;
            for (const auto &s : inconsistencies)
                total_chars += s.size();

            std::string msg("Header is inconsistent at ");
            msg.reserve(msg.size()+total_chars);
            for_each(inconsistencies.begin(), inconsistencies.end(),
                [&msg] (const std::string& s) { msg += s; });

            throw std::runtime_error(msg);
        }

        // Minor
        inconsistencies.clear();
        if (!isClose(hdr.v_length, vlen))
            inconsistencies.push_back("v_length ");
        if (hdr.nvpoints != nvpts)
            inconsistencies.push_back("nvpoints ");
        if (hdr.nzpoints != nzpts)
            inconsistencies.push_back("nzpoints ");
        if (!isClose(hdr.z1, z1))
            inconsistencies.push_back("z1 ");
        if (!isClose(hdr.z_length, zlen))
            inconsistencies.push_back("z_length ");

        if (!inconsistencies.empty()) {
            int total_chars = 1;
            for (const auto &s : inconsistencies)
                total_chars += s.size();

            std::string msg("WARNING: Header is inconsistent at ");
            msg.reserve(msg.size()+total_chars);
            for_each(inconsistencies.begin(), inconsistencies.end(),
                [&msg] (const std::string& s) { msg += s; });

            LOG::LOGGER.ERR(msg.c_str());
        }
    }
}
// ---------------------------------------

/* This function reads following keys from config file:
NumberVPoints: int
NumberZPoints: int
VelocityLength: double
LookUpTableDir: string
    Saves into directory. Default is current dir.
SignalLookUpTableBase: string
    Default is signal.
DerivativeSLookUpTableBase: string
    Default is deriv.
FileNameRList: string
*/
SQLookupTable::SQLookupTable(ConfigFile &config)
{
    N_V_POINTS = config.getInteger("NumberVPoints");
    N_Z_POINTS_OF_S = config.getInteger("NumberZPoints");
    LENGTH_V = config.getDouble("VelocityLength");
    LENGTH_Z_OF_S = bins::Z_BIN_WIDTH * (bins::NUMBER_OF_Z_BINS+1);
    itp_v1 = 0;
    itp_z1 = bins::ZBIN_CENTERS[0] - bins::Z_BIN_WIDTH;

    DIR = config.get("LookUpTableDir", ".");
    S_BASE = config.get("SignalLookUpTableBase", "signal");
    Q_BASE = config.get("DerivativeSLookUpTableBase", "deriv");

    std::string frlist = config.get("FileNameRList");
    if (frlist.empty())
        throw std::invalid_argument("Must pass FileNameRList.");

    NUMBER_OF_R_VALUES = ioh::readListRdv(frlist.c_str(), R_DV_VALUES);

    LOG::LOGGER.STD("Number of R-dv pairs: %d\n", NUMBER_OF_R_VALUES);

    LINEAR_V_ARRAY = sqhelper::allocLinearlySpacedArray(
        itp_v1, LENGTH_V, N_V_POINTS);
    
    if (!specifics::TURN_OFF_SFID)
        LINEAR_Z_ARRAY = sqhelper::allocLinearlySpacedArray(
            itp_z1, LENGTH_Z_OF_S, N_Z_POINTS_OF_S);
}

void SQLookupTable::readTables()
{
    allocateSignalAndDerivArrays();
    LOG::LOGGER.STD("Setting tables.\n");

    if (!specifics::TURN_OFF_SFID)
        interp2d_signal_matrices.reserve(NUMBER_OF_R_VALUES);
    interp_derivative_matrices.reserve(NUMBER_OF_R_VALUES * bins::NUMBER_OF_K_BANDS);

    shared_interp_2d sint;
    std::vector<shared_interp_1d> dints;
    dints.resize(bins::NUMBER_OF_K_BANDS);
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r) {
        _allocReadSQFile(r, sint, dints);
        if (!specifics::TURN_OFF_SFID)
            interp2d_signal_matrices.push_back(sint);

        interp_derivative_matrices.insert(
            interp_derivative_matrices.end(), dints.begin(), dints.end());
    }

    deallocateSignalAndDerivArrays();
}

void SQLookupTable::computeTables(bool force_rewrite)
{
    std::string buf_fnames;
    double time_spent_table_sfid, time_spent_table_q;

    sqhelper::SQ_IO_Header tmp_hdr = {
        N_V_POINTS, N_Z_POINTS_OF_S, LENGTH_V, itp_z1, LENGTH_Z_OF_S, 
        0, 0, bins::K0, bins::DKLIN, bins::DKLOG, bins::NKLIN, bins::NKLOG};

    allocateSignalAndDerivArrays();

    // Initialize loop for parallel computing
    int delta_nr     = NUMBER_OF_R_VALUES / mympi::total_pes, 
        r_start_this = delta_nr * mympi::this_pe, 
        r_end_this   = delta_nr * (mympi::this_pe+1);

    if (mympi::this_pe == mympi::total_pes-1)
        r_end_this = NUMBER_OF_R_VALUES;

    std::unique_ptr<RealField> rft;

    if (!specifics::TURN_OFF_SFID) {
        int nrft = exp2(ceil(log2(2 * N_V_POINTS)) + 1);
        LOG::LOGGER.STD("RealField number of points %d\n", nrft);
        double itp_dv = sqhelper::getLinearSpacing(LENGTH_V, N_V_POINTS);
        rft = std::make_unique<RealField>(nrft, itp_dv / 2);
    }

    for (int r = r_start_this; r < r_end_this; ++r)
    {
        time_spent_table_q = mytime::timer.getTime();

        int Rthis = R_DV_VALUES[r].first;
        double dvthis = R_DV_VALUES[r].second,
               Rkms = SPEED_OF_LIGHT / Rthis / ONE_SIGMA_2_FWHM;

        tmp_hdr.spectrograph_resolution = Rthis;
        tmp_hdr.pixel_width = dvthis;

        LOG::LOGGER.STD(
            "Creating look up tables for derivative matrices:"
            " R=%d (%.2f km/s), dv=%.1f.\n", Rthis, Rkms, dvthis);

        buf_fnames = sqhelper::SQTableFileNameConvention(DIR, S_BASE, Rthis, dvthis);

        if (!force_rewrite && ioh::file_exists(buf_fnames.c_str())) {
            LOG::LOGGER.STD("File %s already exists. Skip to next.\n", buf_fnames.c_str());
            continue;
        }

        buf_fnames.insert(0, "!");
        sqhelper::SQLookupTableFile s_table_file(buf_fnames, true);
        s_table_file.writeMeta(tmp_hdr);

        #pragma omp parallel for
        for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn) {
            double k1 = bins::KBAND_EDGES[kn], k2 = bins::KBAND_EDGES[kn + 1];
            struct spectrograph_windowfn_params win_params = {0, 0, dvthis, Rkms};

            struct sq_integrand_params integration_parameters = {
                &fidpd13::FIDUCIAL_PD13_PARAMS, &win_params};

            FourierIntegrator q_integrator(
                GSL_INTEG_COSINE, q_matrix_integrand, &integration_parameters);

            for (int nv = 0; nv < N_V_POINTS; ++nv) {
                win_params.delta_v_ij = LINEAR_V_ARRAY[nv];

                signal_array[nv + kn * N_V_POINTS] = q_integrator.evaluate(
                    k1, k2, win_params.delta_v_ij, 0);
            }
        }

        s_table_file.writeDeriv(signal_array.get());
        
        time_spent_table_q = mytime::timer.getTime() - time_spent_table_q;
        LOG::LOGGER.STD(
            "Time spent on derivative matrix table R=%d, dv=%.1f is %.2f mins.\n",
            Rthis, dvthis, time_spent_table_q);

        if (specifics::TURN_OFF_SFID)  continue;

        time_spent_table_sfid = mytime::timer.getTime();
        LOG::LOGGER.STD(
            "Creating look up table for signal matrix:"
            " R=%d (%.2f km/s), dv=%.1f.\n",
            Rthis, Rkms, dvthis);

        struct spectrograph_windowfn_params win_params = {
                0, 0, dvthis, Rkms};

        struct sq_integrand_params integration_parameters = {
            &fidpd13::FIDUCIAL_PD13_PARAMS, &win_params};

        for (int nz = 0; nz < N_Z_POINTS_OF_S; ++nz) {
            // z_first + z_length * nz / (Nz - 1.);
            win_params.z_ij = LINEAR_Z_ARRAY[nz];

            for (int i = 0; i < rft->size_k(); ++i)
                rft->field_k[i] = MY_PI * signal_matrix_integrand(
                    rft->k[i], &integration_parameters);

            rft->fftK2X();

            for (int nv = 0; nv < N_V_POINTS; ++nv) {
                int xy = nz + N_Z_POINTS_OF_S * nv;
                signal_array[xy] = rft->field_x[2 * nv];
            }
        }

        s_table_file.writeSignal(signal_array.get());
        time_spent_table_sfid = mytime::timer.getTime() - time_spent_table_sfid;

        LOG::LOGGER.STD(
            "Time spent on fiducial signal matrix table R=%d, dv=%.1f is %.2f mins.\n",
            Rthis, dvthis, time_spent_table_sfid);
    }

    LOG::LOGGER.STD("Deallocating arrays...\n");
    deallocateSignalAndDerivArrays();
}

void SQLookupTable::_allocReadSQFile(
        int r_index, shared_interp_2d& sint,
        std::vector<shared_interp_1d> &dints
) {
    int Rthis = R_DV_VALUES[r_index].first;
    double dvthis = R_DV_VALUES[r_index].second;

    std::string buf_fnames = sqhelper::SQTableFileNameConvention(
        DIR, S_BASE, Rthis, dvthis);
    sqhelper::SQ_IO_Header tmp_hdr;

    LOG::LOGGER.STD("Reading sq_lookup_table_file %s.\n", buf_fnames.c_str());
    sqhelper::SQLookupTableFile s_table_file(buf_fnames, false);

    tmp_hdr = s_table_file.readMeta();
    sqhelper::_checkHeaderConsistency(
        tmp_hdr, N_V_POINTS, N_Z_POINTS_OF_S, 
        LENGTH_V, itp_z1, LENGTH_Z_OF_S, Rthis, dvthis);

    // Allocate memory before reading further
    if (!signal_array)
        allocateSignalAndDerivArrays();

    // Start reading data and set to interp pointer
    if (!specifics::TURN_OFF_SFID)
    {
        s_table_file.readSignal(signal_array.get());
        itp_dv = sqhelper::getLinearSpacing(LENGTH_V, N_V_POINTS);
        itp_dz = sqhelper::getLinearSpacing(LENGTH_Z_OF_S, N_Z_POINTS_OF_S);

        sint = std::make_shared<DiscreteInterpolation2D>(
            itp_z1, itp_dz, itp_v1, itp_dv,
            signal_array.get(), N_Z_POINTS_OF_S, N_V_POINTS);
    }

    s_table_file.readDeriv(signal_array.get());

    itp_dv = sqhelper::getLinearSpacing(LENGTH_V, N_V_POINTS);

    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        dints[kn] = std::make_shared<DiscreteInterpolation1D>(
            itp_v1, itp_dv, N_V_POINTS, signal_array.get() + kn * N_V_POINTS
        );
}


void SQLookupTable::readSQforR(
        int r_index, shared_interp_2d &s,
        std::vector<shared_interp_1d> &q, bool alloc
) {
    if (alloc || interp2d_signal_matrices.empty()) {
        _allocReadSQFile(r_index, s, q);
        return;
    }

    // Skip this section if fiducial signal matrix is turned off.
    if (!specifics::TURN_OFF_SFID)
        s = getSignalMatrixInterp(r_index);

    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
        q[kn] = getDerivativeMatrixInterp(kn, r_index);
}

int SQLookupTable::findSpecResIndex(int spec_res, double dv) const
{
    double rounded_dv = round(dv * 10) / 10;
    for (int r = 0; r < NUMBER_OF_R_VALUES; ++r)
        if ((R_DV_VALUES[r].first == spec_res)
            && (fabs(rounded_dv - R_DV_VALUES[r].second) < 0.01))
            return r;

    return -1;
}

double SQLookupTable::getOneSetMemUsage()
{
    double s = process::getMemoryMB(N_V_POINTS * N_Z_POINTS_OF_S),
           q = process::getMemoryMB(N_V_POINTS * bins::NUMBER_OF_K_BANDS);

    if (specifics::TURN_OFF_SFID) s = 0;

    return s+q;
}

double SQLookupTable::getMaxMemUsage()
{
    double rdvsize = 1.5 * process::getMemoryMB(NUMBER_OF_R_VALUES);

    return getOneSetMemUsage() * NUMBER_OF_R_VALUES + rdvsize;
}

void SQLookupTable::allocateSignalAndDerivArrays()
{
    signal_array = std::make_unique<double[]>(
        N_V_POINTS * std::max(N_Z_POINTS_OF_S, bins::NUMBER_OF_K_BANDS));
}
