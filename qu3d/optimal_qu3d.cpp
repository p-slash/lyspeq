#include <unordered_map>

#include <gsl/gsl_errno.h>

#include "qu3d/optimal_qu3d.hpp"
#include "qu3d/cosmology_3d.hpp"

#include "core/global_numbers.hpp"
#include "core/mpi_manager.hpp"
#include "core/progress.hpp"
#include "mathtools/stats.hpp"
#include "io/logger.hpp"

namespace specifics {
    double MIN_RA = 0, MAX_RA = 0, MIN_DEC = 0, MAX_DEC = 0;
    double MIN_KPERP = 0, MIN_KZ = 0;
}

const double deg2rad = MY_PI / 180.0, rad2deg = 1.0 / deg2rad;
// Assume 2-4 threads will not encounter race conditions
int RINTERP_NTHREADS = 3;


#define KMAX_EDGE bins::KBAND_EDGES[bins::NUMBER_OF_K_BANDS]

/* Timing map */
std::unordered_map<std::string, std::pair<int, double>> timings{
    {"rInterp", std::make_pair(0, 0.0)}, {"interp", std::make_pair(0, 0.0)},
    {"CGD", std::make_pair(0, 0.0)}, {"mDeriv", std::make_pair(0, 0.0)},
    {"mCov", std::make_pair(0, 0.0)}, {"PPcomp", std::make_pair(0, 0.0)},
    {"GenGauss", std::make_pair(0, 0.0)}, {"mIpH", std::make_pair(0, 0.0)},
    {"cgdIpH", std::make_pair(0, 0.0)}, {"Marg", std::make_pair(0, 0.0)}
};

/* Internal variables */
std::unordered_map<size_t, std::vector<const CosmicQuasar*>> idx_quasar_map;

const fidcosmo::FlatLCDM *cosmo;
std::unique_ptr<fidcosmo::ArinyoP3DModel> p3d_model;
std::unique_ptr<ioh::ContMargFile> ioh::continuumMargFileHandler;

int NUMBER_OF_P_BANDS = 0;
double DK_BIN = 0;
bool verbose = true, CONT_MARG_ENABLED = false, KEEP_MATRICES_IN_MEMORY = false;
constexpr bool INPLACE_FFT = true;


#ifdef DEBUG
void CHECK_ISNAN(double *mat, int size, std::string msg) {
    if (std::any_of(mat, mat + size, [](double x) { return std::isnan(x); }))
        throw std::runtime_error(std::string("NAN in ") + msg);
    // std::string line = std::string("No nans in ") + msg + '\n';
    // DEBUG_LOG(line.c_str());
}


void logCosmoDist() {
    DEBUG_LOG("getComovingDist:");
    for (int i = 0; i < 10; ++i) {
        double z1 = 2.95 + i * 0.2;
        DEBUG_LOG(" chi(%.3f) = %.3f", z1 - 1, cosmo->getComovingDist(z1));
    }
    DEBUG_LOG("\n");
}


void logCosmoHubble() {
    DEBUG_LOG("getHubble:");
    for (int i = 0; i < 10; ++i) {
        double z1 = 2.95 + i * 0.2;
        DEBUG_LOG(" H(%.3f) = %.3f", z1 - 1, cosmo->getHubble(z1));
    }
    DEBUG_LOG("\n");
}


void logPmodel() {
    DEBUG_LOG("P3dModel:");
    const double kz = 0.01;
    for (int i = 0; i < 10; ++i) {
        double k = 0.02 + i * 0.02;
        DEBUG_LOG(
            " P3d(kperp=%.2f, kz=%.2f) = %.2f",
            k, kz, p3d_model->interp2d_pL.evaluate(k, kz));
    }
    DEBUG_LOG("\n");
    DEBUG_LOG("VarLss: %.5e\n", p3d_model->getVarLss());
}
#else
#define CHECK_ISNAN(X, Y, Z)
void logCosmoDist() {};
void logCosmoHubble() {};
void logPmodel() {};
#endif


void logTimings() {
    LOG::LOGGER.STD("Total time statistics:\n");
    for (const auto &[key, value] : timings)
        LOG::LOGGER.STD(
            "%s: %.2e mins / %d calls\n",
            key.c_str(), value.second, value.first);
}


inline bool hasConverged(
        double norm, double base, double tolerance, double drate=0
) {
    double rel_norm = norm / base;

    if (verbose)
        LOG::LOGGER.STD(
            "    Current ||r|| / ||b|| = %.3e / %.3e = %.3e. "
            "Conjugate Gradient converges when this is < %.2e\n",
            norm, base, rel_norm, tolerance);

    if (verbose && drate > 0)
        LOG::LOGGER.STD("    Smoothed descent rate is %.3f.\n", drate);

    return (norm < tolerance) || (rel_norm < tolerance);
}


void _shiftByMedianDecRa(std::vector<std::unique_ptr<CosmicQuasar>> &quasars) {
    std::vector<double> thetas, phis;
    thetas.reserve(quasars.size());
    phis.reserve(quasars.size());

    for (const auto &qso : quasars) {
        phis.push_back(qso->vec.phi);
        thetas.push_back(qso->vec.theta);
    }

    double median_phi = stats::medianOfUnsortedVector(phis),
           median_theta = stats::medianOfUnsortedVector(thetas);

    LOG::LOGGER.STD(
        "Rotating to the median pointing (theta:%.4f, phi:%.4f) deg\n",
        median_theta * rad2deg, median_phi * rad2deg);

    const auto rot_mat = Vec3::getRotationMatrix(Vec3(median_theta, median_phi));
    for (auto &qso : quasars)
        #ifdef USE_SPHERICAL_DIST
        qso->vec.rotate(rot_mat);
        #else
        qso->vec.setAngles(qso->vec.theta - median_theta, qso->vec.phi - median_phi);
        #endif
}


double _setCosmologicalCoordinates(
    std::vector<std::unique_ptr<CosmicQuasar>> &quasars
) {
    double sum_chi_weights = 0, sum_weights = 0;

    #pragma omp parallel for num_threads(8) reduction(+:sum_chi_weights, sum_weights)
    for (const auto &qso : quasars)
        qso->getSumRadialDistance(cosmo, sum_chi_weights, sum_weights);

    sum_chi_weights /= sum_weights;

    LOG::LOGGER.STD("Effective radial distance is %.3f Mpc/h.\n", sum_chi_weights);

    #pragma omp parallel for num_threads(8)
    for (auto &qso : quasars)
        qso->setComovingDistances(cosmo, sum_chi_weights);

    return sum_chi_weights;
}

void _setSpectroMeanParams(
        const std::vector<std::unique_ptr<CosmicQuasar>> &quasars
) {
    double mean_sigma = 0, mean_delta_r = 0;
    for (const auto &qso : quasars) {
        double sigma, delta_r;
        qso->getSpectroWindowParams(cosmo, sigma, delta_r);
        mean_sigma += sigma;
        mean_delta_r += delta_r;
    }

    mean_sigma /= quasars.size();
    mean_delta_r /= quasars.size();

    LOG::LOGGER.STD(
        "Mean spectro window params: s=%.2f Mpc/h and Delta r=%.2f Mpc/h.\n",
        mean_sigma, mean_delta_r);
    p3d_model->setSpectroParams(mean_sigma, mean_delta_r);
}


void Qu3DEstimator::_readOneDeltaFile(const std::string &fname) {
    qio::PiccaFile pFile(fname);
    int number_of_spectra = pFile.getNumberSpectra();
    std::vector<std::unique_ptr<CosmicQuasar>> local_quasars;
    local_quasars.reserve(number_of_spectra);

    for (int i = 0; i < number_of_spectra; ++i) {
        try {
            local_quasars.push_back(
                std::make_unique<CosmicQuasar>(&pFile, i + 1));
        } catch (std::exception& e) {
            #ifdef VERBOSE
            std::ostringstream fpath;
            fpath << fname << '[' << i + 1 << ']';
            LOG::LOGGER.ERR(
                "%s. Filename %s.\n", e.what(), fpath.str().c_str());
            #endif
        }
    }

    pFile.closeFile();

    if (local_quasars.empty())
        return;

    #pragma omp critical
    {
        quasars.reserve(quasars.size() + local_quasars.size());
        std::move(std::begin(local_quasars), std::end(local_quasars),
                  std::back_inserter(quasars));
    }
}


void Qu3DEstimator::_readQSOFiles(
        const std::string &flist, const std::string &findir
) {
    double t1 = mytime::timer.getTime(), t2 = 0;
    std::vector<std::string> filepaths;

    LOG::LOGGER.STD("Read delta files.\n");
    qio::PiccaFile::use_cache = false;

    int number_of_files = ioh::readList(flist.c_str(), filepaths);

    std::vector<long> targetid_to_remove;
    std::string fname_t2ig = config.get("Targetids2Ignore");
    if (!fname_t2ig.empty()) {
        ioh::readList(fname_t2ig.c_str(), targetid_to_remove, false);
        LOG::LOGGER.STD("Reading TARGETIDs to remove. There are %zu items.\n",
                        targetid_to_remove.size());
    }

    if (mympi::total_pes > 1) {
        int nrot = mympi::this_pe * number_of_files / mympi::total_pes;
        std::rotate(filepaths.begin(), filepaths.begin() + nrot,
                    filepaths.end());
    }

    #pragma omp parallel for num_threads(8)
    for (auto &fq : filepaths) {
        fq.insert(0, findir);  // Add parent directory to file path
        _readOneDeltaFile(fq);
    }

    if (quasars.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");

    if (!targetid_to_remove.empty()) {
        int nerased = std::erase_if(quasars, [&targetid_to_remove](auto &qso) {
            long id = qso->qFile->id;
            auto it = std::find(
                targetid_to_remove.cbegin(), targetid_to_remove.cend(), id);
            return it != targetid_to_remove.cend();
        });
        LOG::LOGGER.STD("Removed %d quasars using TARGETID list.\n", nerased);
    }

    _shiftByMedianDecRa(quasars);
    effective_chi = _setCosmologicalCoordinates(quasars);
    _setSpectroMeanParams(quasars);

    t2 = mytime::timer.getTime();

    int max_qN = 0;
    #pragma omp parallel for reduction(+:num_all_pixels) reduction(max:max_qN)
    for (auto &qso : quasars) {
        // Dropping pixels with ivar=0 after setting spectro parameters may
        // help with better estimation of spectro window parameters. Also,
        // these pixels do not contribute to the signal and may cause numerical
        // issues in some cases.
        if (!test_gaussian_field)
            qso->dropIvar0Pixels();
        num_all_pixels += qso->N;
        max_qN = std::max(qso->N, max_qN);
    }

    CosmicQuasar::allocCcov(max_qN * max_qN);
    CosmicQuasar::allocRrmat(max_qN * max_qN);

    LOG::LOGGER.STD(
        "There are %d quasars and %ld number of pixels. "
        "Reading QSO files took %.2f m.\n",
        quasars.size(), num_all_pixels, t2 - t1);
}


void Qu3DEstimator::_calculateBoxDimensions(float L[3], float xyz0[3]) {
    float lxmin = 1e15, lymin = 1e15, lzmin = 1e15,
          lxmax = -1e15, lymax = -1e15, lzmax = -1e15;

    #pragma omp parallel for reduction(min:lxmin, lymin, lzmin) \
                             reduction(max:lxmax, lymax, lzmax)
    for (auto it = quasars.cbegin(); it != quasars.cend(); ++it) {
        const CosmicQuasar *qso = it->get();        
        lxmin = std::min(lxmin, std::min(qso->r[0], qso->r[3 * qso->N - 3]));
        lxmax = std::max(lxmax, std::max(qso->r[0], qso->r[3 * qso->N - 3]));

        lymin = std::min(lymin, std::min(qso->r[1], qso->r[3 * qso->N - 2]));
        lymax = std::max(lymax, std::max(qso->r[1], qso->r[3 * qso->N - 2]));

        lzmin = std::min(lzmin, std::min(qso->r[2], qso->r[3 * qso->N - 1]));
        lzmax = std::max(lzmax, std::max(qso->r[2], qso->r[3 * qso->N - 1]));
    }

    L[0] = lxmax - lxmin;
    L[1] = lymax - lymin;
    L[2] = lzmax - lzmin;
    xyz0[0] = lxmin;
    xyz0[1] = lymin;
    xyz0[2] = lzmin;
}


void Qu3DEstimator::_setupMesh(double radius, double minboxlength) {
    double t1 = mytime::timer.getTime(), t2 = 0, extra_l = 0;

    _calculateBoxDimensions(mesh.length, mesh.xyz0);
    LOG::LOGGER.STD(
        "Initial box dimensions are as follows: "
        "L = (%.0f, %.0f, %.0f) Mpc/h, XYZ0 = (%.0f, %.0f, %.0f) Mpc/h.\n",
        mesh.length[0], mesh.length[1], mesh.length[2],
        mesh.xyz0[0], mesh.xyz0[1], mesh.xyz0[2]);

    mesh.ngrid[0] = config.getInteger("NGRID_X");
    mesh.ngrid[1] = config.getInteger("NGRID_Y");
    mesh.ngrid[2] = config.getInteger("NGRID_Z");

    auto padMesh = [this](double dL, int a) {
        mesh.length[a] += dL;
        mesh.xyz0[a] -= dL / 2.0;
    };

    double dx = mesh.length[0] / mesh.ngrid[0],
           dy = mesh.length[1] / mesh.ngrid[1],
           dz = mesh.length[2] / mesh.ngrid[2];

    #ifdef USE_SPHERICAL_DIST
    if (true) {
    #else
    double delta_rad = specifics::MAX_RA - specifics::MIN_RA - 2 * MY_PI;
    if (fabs(delta_rad) > (2 * MY_PI * DOUBLE_EPSILON)) {
    #endif
        mesh.disablePeriodicityX();
        padMesh(std::max(10.0 * dx, minboxlength - mesh.length[0]), 0);
    }

    padMesh(std::max(10.0 * dy, minboxlength - mesh.length[1]), 1);
    padMesh(std::max(10.0 * dz, minboxlength - mesh.length[2]), 2);
    dx = mesh.length[0] / mesh.ngrid[0];
    dy = mesh.length[1] / mesh.ngrid[1];

    double dyl = dx - dy;
    if (dyl > 0) {
        dyl *= mesh.ngrid[1];
        padMesh(dyl, 1);
    }
    else {
        dyl = fabs(dyl) * mesh.ngrid[0];
        padMesh(dyl, 0);
    }

    if (config.getInteger("MatchCellSizeOfZToXY") > 0) {
        dx = mesh.length[0] / mesh.ngrid[0];
        dy = mesh.length[1] / mesh.ngrid[1];
        dz = mesh.length[2] / mesh.ngrid[2];
        double dzl = (dx + dy) / 2 - dz;

        if (dzl > 0) {
            dzl *= mesh.ngrid[2];
            LOG::LOGGER.STD(
                "Automatically padding z axis to match cell length in x & y "
                "directions by %.3f /h.\n", dzl);
            padMesh(dzl, 2);
        }
    }
    
    LOG::LOGGER.STD(
        "Final box dimensions are as follows: "
        "L = (%.0f, %.0f, %.0f) Mpc/h, XYZ0 = (%.0f, %.0f, %.0f) Mpc/h.\n",
        mesh.length[0], mesh.length[1], mesh.length[2],
        mesh.xyz0[0], mesh.xyz0[1], mesh.xyz0[2]);

    mesh.construct(INPLACE_FFT);

    LOG::LOGGER.STD("Mesh cell dimensions are as follows: "
                    "dx = (%.3f, %.3f, %.3f) Mpc/h.\n",
                    mesh.dx[0], mesh.dx[1], mesh.dx[2]);

    // Shift coordinates of quasars
    LOG::LOGGER.STD("Shifting quasar locations to center the mesh.\n");
    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->r[0 + 3 * i] -= mesh.xyz0[0];
            qso->r[1 + 3 * i] -= mesh.xyz0[1];
            qso->r[2 + 3 * i] -= mesh.xyz0[2];
        }
    }

    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("Mesh construct took %.2f m.\n", t2 - t1);
}


void Qu3DEstimator::_constructMap() {
    double t1 = mytime::timer.getTime(), t2 = 0;

    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->findGridPoints(mesh);

    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("findGridPoints took %.2f m.\n", t2 - t1);

    LOG::LOGGER.STD("Sorting quasars by mininum NGP index.\n");
    std::sort(
        quasars.begin(), quasars.end(), [](const auto &q1, const auto &q2) {
            if (q1->min_x_idx == q2->min_x_idx)
                return q1->qFile->id < q2->qFile->id;

            return q1->min_x_idx < q2->min_x_idx; }
    );

    for (const auto &qso : quasars)
        for (const auto &i : qso->grid_indices)
            idx_quasar_map[i].push_back(qso.get());

    t1 = mytime::timer.getTime();
    LOG::LOGGER.STD("Appending map took %.2f m.\n", t1 - t2);
}


void Qu3DEstimator::_findNeighbors() {
    /* Assumes radius is multiplied by factor. */
    double t1 = mytime::timer.getTime(), t2 = 0;
    double radius2 = radius * radius;

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        auto neighboring_pixels = qso->findNeighborPixels(mesh, radius);

        for (const size_t &ipix : neighboring_pixels) {
            auto kumap_itr = idx_quasar_map.find(ipix);

            if (kumap_itr == idx_quasar_map.end())
                continue;

            qso->neighbors.insert(
                kumap_itr->second.cbegin(), kumap_itr->second.cend());
        }

        /* Check if self is in. Should be.
        long id = qso->qFile->id;
        const auto it = std::find_if(
            qso->neighbors.cbegin(), qso->neighbors.cend(),
            [&id](const auto &q) { return id == q->qFile->id; }
        );

        if (it == qso->neighbors.cend())
            LOG::LOGGER.STD("Self not in\n");
        if (qso->neighbors.contains(qso.get()))
            LOG::LOGGER.STD("Self is in\n"); */

        qso->trimNeighbors(radius2, effective_chi);
    }

    // symmetrize neighbors. Should not erase anything.
    double mean_num_neighbors = 0;
    for (auto &qso : quasars) {
        CosmicQuasar* qso_ptr = qso.get();
        int num_erased = std::erase_if(
            qso->neighbors, [&qso_ptr](const CosmicQuasar* const &q) {
                return !q->neighbors.contains(qso_ptr);
        });

        if (num_erased > 0)  LOG::LOGGER.STD("WARNING: Erased neighbors.\n");
        mean_num_neighbors += qso->neighbors.size();
    }

    mean_num_neighbors /= quasars.size();
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD(
        "_findNeighbors took %.2f m. Average number of neighbors: %.3f\n",
        t2 - t1, mean_num_neighbors);

    if (mean_num_neighbors == 0)
        throw std::runtime_error("No neighbors detected even though PP is enabled.");

    _saveNeighbors();
}


void Qu3DEstimator::_saveNeighbors() {
    if (mympi::this_pe != 0) {
        mympi::barrier();  return;
    }

    int status = 0;

    std::string out_fname = "!" + process::FNAME_BASE + "-quasar-neighbors.fits";
    auto fitsfile_ptr = ioh::create_unique_fitsfile_ptr(out_fname);
    fitsfile *fits_file = fitsfile_ptr.get();

    /* define the name, datatype, and physical units for columns */
    int ncolumns = 2;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wwrite-strings"
    char *column_names[] = {"TARGETID", "NEIGHBORS"};
    char *column_types[] = {"1K", "1PK"};
    char *column_units[] = { "", ""};
    #pragma GCC diagnostic pop

    fits_create_tbl(fits_file, BINARY_TBL, quasars.size(), ncolumns,
                    column_names, column_types, column_units, "NEIGHBORS",
                    &status);
    ioh::checkFitsStatus(status);

    // fits_write_key(
    //     fits_file, TDOUBLE, "RA", &qso->angles[0], nullptr, &status);
    // fits_write_key(
    //     fits_file, TDOUBLE, "DEC", &qso->angles[1], nullptr, &status);
    // fits_write_key(
    //     fits_file, TDOUBLE, "MEAN_SNR", &qso->qFile->snr, nullptr, &status);
    // fits_write_key(fits_file, TINT, "NUM_NEIG", &nmbrs, nullptr, &status);

    int irow = 1;
    for (const auto &qso : quasars) {
        int n = qso->neighbors.size(), i = 0;
        if (n == 0)
            continue;
        auto nbrs = std::make_unique<long[]>(n);
        for (const CosmicQuasar* q : qso->neighbors) {
            nbrs[i] = q->qFile->id;  ++i;
        }
        fits_write_col(fits_file, TLONGLONG, 1, irow, 1, 1, &qso->qFile->id, &status);
        fits_write_col(fits_file, TLONGLONG, 2, irow, 1, n, nbrs.get(), &status);
        ioh::checkFitsStatus(status);
        ++irow;
    }

    mympi::barrier();
    LOG::LOGGER.STD("Neighbors cache saved as %s\n", out_fname.c_str());
}

void Qu3DEstimator::_readNeighbors(const std::string &neighbors_file) {
    double t1 = mytime::timer.getTime(), t2 = 0;
    int status = 0, hdutype;
    fitsfile *fits_file = nullptr;
    long nrows;

    std::unordered_map<long, const CosmicQuasar*> targetid_pointer_map;
    std::unordered_map<long, std::vector<long>> targetid_neighbors_map;
    for (const auto &qso : quasars)
        targetid_pointer_map[qso->qFile->id] = qso.get();

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wwrite-strings"
    fits_open_extlist(&fits_file, neighbors_file.c_str(), READONLY,
                     "NEIGHBORS", &hdutype, &status);
    #pragma GCC diagnostic pop

    ioh::checkFitsStatus(status);
    if (hdutype != BINARY_TBL)
        throw std::runtime_error("HDU type is not BINARY!");

    fits_get_num_rows(fits_file, &nrows, &status);
    for (long i = 1; i <= nrows; ++i) {
        long nelem = 0, offset, targetid;
        fits_read_col(fits_file, TLONGLONG, 1, i, 1, 1,
                      nullptr, &targetid, nullptr, &status);
        fits_read_descript(fits_file, 2, i, &nelem, &offset, &status);
        // printf("targetid, nelem, offset: %ld, %ld, %ld\n",
        //        targetid, nelem, offset);
        if (nelem == 0)  continue;

        targetid_neighbors_map[targetid].resize(nelem);
        fits_read_col(fits_file, TLONGLONG, 2, i, 1, nelem,
                      nullptr, targetid_neighbors_map[targetid].data(),
                      nullptr, &status);
    }
    fits_close_file(fits_file, &status);
    ioh::checkFitsStatus(status);

    double mean_num_neighbors = 0;
    for (auto &qso : quasars) {
        for (const long &t : targetid_neighbors_map[qso->qFile->id])
            qso->neighbors.insert(targetid_pointer_map[t]);
        mean_num_neighbors += qso->neighbors.size();
    }

    mean_num_neighbors /= quasars.size();
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD(
        "_readNeighbors took %.2f m. Average number of neighbors: %.3f\n",
        t2 - t1, mean_num_neighbors);
}


void Qu3DEstimator::_createRmatFiles(const std::string &prefix) {
    /* This function needs z1 to be 1 + z */
    double t1 = mytime::timer.getTime(), t2 = 0;
    LOG::LOGGER.STD("Calculating R_D matrices for continuum marginalization. ");
    ioh::continuumMargFileHandler = std::make_unique<ioh::ContMargFile>(
            process::TMP_FOLDER, prefix);

    std::vector<int> q_fidx;
    std::vector<long> q_ids;
    size_t nquasars = quasars.size();

    if (mympi::this_pe == 0) {
        ioh::continuumMargFileHandler->openAllWriters();

        #pragma omp parallel for schedule(static, 8)
        for (auto &qso : quasars)
            qso->constructMarginalization(specifics::CONT_LOGLAM_MARG_ORDER,
                                          KEEP_MATRICES_IN_MEMORY);

        ioh::continuumMargFileHandler->closeAllWriters();

        q_fidx.reserve(nquasars);  q_ids.reserve(nquasars);
        for (auto it = quasars.cbegin(); it != quasars.cend(); ++it) {
            q_fidx.push_back((*it)->fidx);  q_ids.push_back((*it)->qFile->id);
        }
    }
    else {
        q_fidx.resize(nquasars);  q_ids.resize(nquasars);
    }

    mympi::barrier();
    mympi::bcast(q_fidx.data(), nquasars);
    mympi::bcast(q_ids.data(), nquasars);
    for (size_t i = 0; i < nquasars; ++i) {
        if (quasars[i]->qFile->id != q_ids[i])
            LOG::LOGGER.ERR("Quasars do no align!!!\n");
        quasars[i]->fidx = q_fidx[i];
    }

    if (!KEEP_MATRICES_IN_MEMORY)
        ioh::continuumMargFileHandler->openAllReaders();
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("It took %.2f m.\n", t2 - t1);
}


void Qu3DEstimator::_openResultsFile() {
    result_file = std::make_unique<ioh::Qu3dFile>(
        process::FNAME_BASE, mympi::this_pe);

    p3d_model->write(result_file.get());

    double *k_grid = raw_power.get(),
           *pfid_grid = raw_bias.get();

    for (int imu = 0; imu < number_of_multipoles; ++imu) {
        for (int ik = 0; ik < bins::NUMBER_OF_K_BANDS; ++ik) {
            size_t i = ik + bins::NUMBER_OF_K_BANDS * imu;
            k_grid[i] = bins::KBAND_CENTERS[ik];
            pfid_grid[i] = p3d_model->evalP3dL(k_grid[i], imu);
        }
    }

    result_file->write(k_grid, NUMBER_OF_P_BANDS, "K");
    result_file->write(pfid_grid, NUMBER_OF_P_BANDS, "PFID");
    result_file->flush();
    std::fill_n(k_grid, NUMBER_OF_P_BANDS, 0);
    std::fill_n(pfid_grid, NUMBER_OF_P_BANDS, 0);
}


Qu3DEstimator::Qu3DEstimator(ConfigFile &configg) : config(configg) {
    config.addDefaults(qu3d_default_parameters);

    specifics::MIN_RA = config.getDouble("MinimumRa") * deg2rad;
    specifics::MAX_RA = config.getDouble("MaximumRa") * deg2rad;
    specifics::MIN_DEC = config.getDouble("MinimumDec") * deg2rad;
    specifics::MAX_DEC = config.getDouble("MaximumDec") * deg2rad;
    specifics::MIN_KPERP = config.getDouble("MinimumKperp");
    specifics::MIN_KZ = std::max(0.0, config.getDouble("MinimumKlos"));
    double minboxlength = config.getDouble("MinBoxLength");

    LOG::LOGGER.STD(
        "Sky cut: RA %.3f-%.3f & DEC %.3f-%.3f in radians.\n",
        specifics::MIN_RA, specifics::MAX_RA,
        specifics::MIN_DEC, specifics::MAX_DEC);

    num_all_pixels = 0;
    std::string
        flist = config.get("FileNameList"),
        findir = config.get("FileInputDir"),
        seed = config.get("Seed") + std::to_string(mympi::this_pe),
        unique_prefix = config.get("UniquePrefixTmp");

    if (flist.empty())
        throw std::invalid_argument("Must pass FileNameList.");
    if (findir.empty())
        throw std::invalid_argument("Must pass FileInputDir.");

    if (findir.back() != '/')
        findir += '/';

    pp_enabled = config.getInteger("TurnOnPpCovariance") > 0;
    max_conj_grad_steps = config.getInteger("MaxConjGradSteps");
    predeconvolve_cic_window = config.getInteger("DeconvolveCICWindow") > 0;
    int use_tsc_interpolation = config.getInteger("UseTscInterpolation") > 0;
    RINTERP_NTHREADS = config.getInteger("ReverseInterpolationThreadNum");
    max_monte_carlos = config.getInteger("MaxMonteCarlos");
    mock_grid_res_factor = config.getInteger("MockGridResolutionFactor");
    test_gaussian_field = config.getInteger("TestGaussianField") > 0;
    pade_order = config.getInteger("PadeOrder");
    tolerance = config.getDouble("ConvergenceTolerance");
    mc_tol = tolerance;
    absolute_tolerance = config.getInteger("AbsoluteTolerance") > 0;
    specifics::DOWNSAMPLE_FACTOR = config.getInteger("DownsampleFactor");
    radius = config.getDouble("LongScale");
    rscale_factor = config.getDouble("ScaleFactor");
    mixture_factor_for_as = std::clamp(
        config.getDouble("MixtureFactorForAs"), 0.0, 1.0);

    // if (rscale_factor > fidcosmo::ArinyoP3DModel::MAX_R_FACTOR)
    //     throw std::invalid_argument(
    //         "ScaleFactor cannot exceed "
    //         + std::to_string(fidcosmo::ArinyoP3DModel::MAX_R_FACTOR));

    total_bias_enabled = config.getInteger("EstimateTotalBias") > 0;
    total_bias_direct_enabled = config.getInteger("EstimateTotalBiasDirectly") > 0;
    noise_bias_enabled = config.getInteger("EstimateNoiseBias") > 0;
    fisher_direct_enabled = config.getInteger("EstimateFisherDirectly") > 0;
    max_eval_enabled = config.getInteger("EstimateMaxEigenValues") > 0;
    number_of_multipoles = config.getInteger("NumberOfMultipoles");
    shrink_factor_for_sqrt = config.getDouble("ShrinkFactorForSqrt");
    CONT_MARG_ENABLED = specifics::CONT_LOGLAM_MARG_ORDER > -1;
    KEEP_MATRICES_IN_MEMORY = config.getInteger("KeepMatricesInMemory") > 0;

    if (CONT_MARG_ENABLED && unique_prefix.empty())
        throw std::invalid_argument("Need UniquePrefixTmp when marginalizing.");

    seed_generator = std::make_unique<std::seed_seq>(seed.begin(), seed.end());
    _initRngs(seed_generator.get());

    p3d_model = std::make_unique<fidcosmo::ArinyoP3DModel>(config);
    cosmo = p3d_model->getCosmoPtr();
    logCosmoDist(); logCosmoHubble(); 

    NUMBER_OF_P_BANDS = bins::NUMBER_OF_K_BANDS * number_of_multipoles;
    DK_BIN = bins::KBAND_CENTERS[1] - bins::KBAND_CENTERS[0];
    bins::FISHER_SIZE = NUMBER_OF_P_BANDS * NUMBER_OF_P_BANDS;

    raw_power = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    filt_power = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    raw_bias = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    filt_bias = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    fisher = std::make_unique<double[]>(bins::FISHER_SIZE);
    covariance = std::make_unique<double[]>(bins::FISHER_SIZE);

    _readQSOFiles(flist, findir);

    LOG::LOGGER.STD("Calculating cosmology model.\n");
    p3d_model->construct();
    p3d_model->calcVarLss(pp_enabled);
    LOG::LOGGER.STD("VarLSS: %.3e.\n", p3d_model->getVarLss());
    logPmodel();

    _openResultsFile();

    if (use_tsc_interpolation)
        mesh.useTscInterpolation();

    _setupMesh(radius, minboxlength);
    _constructMap();
    radius *= rscale_factor;

    std::string neighbors_file = config.get("NeighborsCache");
    if (pp_enabled) {
        if (neighbors_file.empty())  _findNeighbors();
        else  _readNeighbors(neighbors_file);
    }

    if (KEEP_MATRICES_IN_MEMORY)
        for (auto &qso : quasars)
            qso->allocMore();

    bool end_imm = (test_gaussian_field && (mock_grid_res_factor > 1));
    if (CONT_MARG_ENABLED && !end_imm)
        _createRmatFiles(unique_prefix);

    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->transformZ1toG(p3d_model.get());
}

void Qu3DEstimator::reverseInterpolate(RealField3D &m) {
    double dt = mytime::timer.getTime();
    m.zero_field_x();

    #pragma omp parallel for num_threads(RINTERP_NTHREADS)
    for (const auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            m.reverseInterpolate(qso->r.get() + 3 * i, qso->in[i]);
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["rInterp"].first;
    timings["rInterp"].second += dt;
}


void Qu3DEstimator::reverseInterpolateZ(RealField3D &m) {
    double dt = mytime::timer.getTime();
    m.zero_field_x();

    #pragma omp parallel for num_threads(RINTERP_NTHREADS)
    for (const auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            m.reverseInterpolate(
                qso->r.get() + 3 * i, qso->in[i] * qso->z1[i]);
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["rInterp"].first;
    timings["rInterp"].second += dt;
}


void Qu3DEstimator::reverseInterpolateIsig(RealField3D &m) {
    double dt = mytime::timer.getTime();
    m.zero_field_x();

    #pragma omp parallel for num_threads(RINTERP_NTHREADS)
    for (const auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            m.reverseInterpolate(qso->r.get() + 3 * i, qso->in_isig[i]);
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["rInterp"].first;
    timings["rInterp"].second += dt;
}


void Qu3DEstimator::multMeshComp() {
    double t1 = mytime::timer.getTime(), t2 = 0;

    reverseInterpolateIsig(mesh);
    mesh.convolvePk(p3d_model->interp2d_pL, predeconvolve_cic_window);

    double dt = mytime::timer.getTime();
    // Interpolate and Weight by isig
    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->interpMesh2Out(mesh);

    t2 = mytime::timer.getTime();
    ++timings["interp"].first;
    timings["interp"].second += t2 - dt;

    if (verbose)
        LOG::LOGGER.STD("    multMeshComp took %.2f s.\n", 60.0 * (t2 - t1));
}


void Qu3DEstimator::multParticleComp(bool neighbors_only) {
    double t1 = mytime::timer.getTime(), dt = 0;

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        qso->multCovNeighbors(p3d_model.get(), effective_chi, !neighbors_only);

    dt = mytime::timer.getTime() - t1;
    ++timings["PPcomp"].first;
    timings["PPcomp"].second += dt;

    if (verbose)
        LOG::LOGGER.STD("    multParticleComp took %.2f s.\n", 60.0 * dt);
}

void Qu3DEstimator::multiplyCovVector(bool mesh_enabled) {
    /* Multiply each quasar's *in pointer and save to *out pointer.
       (I + R^-1/2 N^-1/2 G^1/2 S G^1/2 N^-1/2 R^-1/2) z = out
    */
    double dt = mytime::timer.getTime();

    // Multiply out with marg. matrix if enabled
    // Multiply out with isig
    // Evolve with redshift growth
    if (CONT_MARG_ENABLED) {
        #pragma omp parallel for schedule(static, 8)
        for (auto &qso : quasars)
            qso->setInIsigWithMarg();

        ioh::continuumMargFileHandler->rewind();
        ++timings["Marg"].first;
        timings["Marg"].second += mytime::timer.getTime() - dt;
    }
    else {
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->setInIsigNoMarg();
    }

    // Add long wavelength mode to Cy
    if (mesh_enabled) {
        multMeshComp();
    }
    else {
        #pragma omp parallel for
        for (auto &qso : quasars)
            std::fill_n(qso->out, qso->N, 0);
    }

    if (pp_enabled)
        multParticleComp();

    // Evolve out with redshift growth
    // Multiply out with isig (These are saved to z1)
    // Multiply out with marg. matrix if enabled
    // Add I.y to out
    if (CONT_MARG_ENABLED) {
        double tm1 = mytime::timer.getTime();

        #pragma omp parallel for schedule(static, 8)
        for (auto &qso : quasars) {
            for (int i = 0; i < qso->N; ++i)
                qso->out[i] *= qso->isig[i] * qso->z1[i];

            qso->multInputWithMarg(qso->out);
            std::swap(qso->out, qso->in_isig);

            #pragma omp simd
            for (int i = 0; i < qso->N; ++i)
                qso->out[i] += qso->in[i];
        }
        ioh::continuumMargFileHandler->rewind();
        ++timings["Marg"].first;
        timings["Marg"].second += mytime::timer.getTime() - tm1;
    }
    else {
        #pragma omp parallel for
        for (auto &qso : quasars) {
            for (int i = 0; i < qso->N; ++i) {
                qso->out[i] *= qso->isig[i] * qso->z1[i];
                qso->out[i] += qso->in[i];
            }
        }
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["mCov"].first;
    timings["mCov"].second += dt;
}


double Qu3DEstimator::updateY(double residual_norm2) {
    double t1 = mytime::timer.getTime(), t2 = 0;

    double pTCp = 0, norm_p = 0, norm_Cp = 0,
           alpha = 0, new_residual_norm = 0;

    /* Multiply C x search into Cy => C. p(in) = out*/
    updateYMatrixVectorFunction();

    // Get pT . C . p
    #pragma omp parallel for reduction(+:pTCp, norm_p, norm_Cp)
    for (const auto &qso : quasars) {
        norm_p += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
        norm_Cp += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);
        pTCp += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);
    }
    norm_p *= norm_Cp * 1e-14;
    norm_p = sqrt(norm_p);

    if (pTCp <= 0) {
        LOG::LOGGER.ERR("Negative pTCp = %.9e (All), ", pTCp);
        dumpSearchDirection();

        pTCp = 0;
        verbose = false;
        multMeshComp();
        #pragma omp parallel for reduction(+:pTCp, alpha)
        for (auto &qso : quasars) {
            for (int i = 0; i < qso->N; ++i)
                qso->out[i] *= qso->isig[i] * qso->z1[i];
            pTCp += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);
            alpha += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
        }

        LOG::LOGGER.ERR("pTp = %.9e, pTS_Lp = %.9e, ", alpha, pTCp);
        pTCp = 0;
        #pragma omp parallel for schedule(dynamic, 4) reduction(+:pTCp)
        for (auto &qso : quasars) {
            std::fill_n(qso->out, qso->N, 0);
            qso->multCovNeighbors(p3d_model.get(), effective_chi);
            for (int i = 0; i < qso->N; ++i)
                qso->out[i] *= qso->isig[i] * qso->z1[i];
            pTCp += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);
        }

        LOG::LOGGER.ERR("pTS_Sp = %.9e.\n", pTCp);
        return 0;
    }

    if (pTCp < norm_p)  return -1.0;

    alpha = residual_norm2 / pTCp;

    /* Update y in the search direction, restore qso->in
       Update residual */
    #pragma omp parallel for reduction(+:new_residual_norm)
    for (auto &qso : quasars) {
        /* in is search.get() */
        cblas_daxpy(qso->N, alpha, qso->in, 1, qso->y.get(), 1);
        cblas_daxpy(qso->N, -alpha, qso->out, 1, qso->residual.get(), 1);

        new_residual_norm += cblas_ddot(
            qso->N, qso->residual.get(), 1, qso->residual.get(), 1);
    }

    t2 = mytime::timer.getTime() - t1;
    if (verbose)
        LOG::LOGGER.STD("    updateY took %.2f s.\n", 60.0 * t2);

    return sqrt(new_residual_norm);
}


void Qu3DEstimator::preconditionerSolution() {
    double dt = mytime::timer.getTime();

    if (CONT_MARG_ENABLED) {
        /* Marginalize, initial guess, marginalize */
        #pragma omp parallel for schedule(static, 8)
        for (auto &qso : quasars) {
            double *rrmat = qso->multInputWithMarg(qso->truth);
            qso->multInvCov(p3d_model.get(), qso->in_isig, qso->truth);
            cblas_dsymv(CblasRowMajor, CblasUpper, qso->N, 1.0,
                        rrmat, qso->N, qso->truth, 1, 0, qso->in, 1);
            qso->multIsigInVector();
        }
        ioh::continuumMargFileHandler->rewind();
        ++timings["Marg"].first;
        timings["Marg"].second += mytime::timer.getTime() - dt;
    }
    else {
        #pragma omp parallel for schedule(dynamic, 8)
        for (auto &qso : quasars) {
            qso->multInvCov(p3d_model.get(), qso->truth, qso->in);
            qso->multIsigInVector();
        }
    }

    ++timings["CGD"].first;
    timings["CGD"].second += mytime::timer.getTime() - dt;
}


void Qu3DEstimator::conjugateGradientDescent() {
    if (max_conj_grad_steps <= 0) {
        preconditionerSolution();
        return;
    }

    static auto convergence_file = std::make_unique<ioh::Qu3dFile>(
        process::FNAME_BASE + "-convergence-" + std::to_string(mympi::this_pe),
        0);

    double dt = mytime::timer.getTime();
    int niter = 1;
    bool restart = false;

    double init_residual_norm = 0, old_residual_prec = 0,
           new_residual_norm = 0, truth_norm = 0;

    std::vector<double> conv_vec;
    conv_vec.reserve(max_conj_grad_steps + 1);
    updateYMatrixVectorFunction = [this]() { this->multiplyCovVector(); };

    auto calculateDescentRate = [&conv_vec](int window=10) {
        if (conv_vec.size() < 2) return 0.0;

        size_t i0 = (conv_vec.size() > window) ? conv_vec.size() - window : 1;

        double sum = 0.0, weights = 0.0;
        for (size_t i = i0; i < conv_vec.size(); ++i) {
            if (conv_vec[i] < 0)  continue;
            double w = (i - i0 + 1.0);
            sum += exp(-log(conv_vec[i]) / i) * w;
            weights += w;
        }
        return sum / weights;
    };

    if (verbose)
        LOG::LOGGER.STD("  Entered conjugateGradientDescent.\n");
    
    std::function<void(std::unique_ptr<CosmicQuasar> &qso,
                const double *input, double *output)> preconditioner;
    if (KEEP_MATRICES_IN_MEMORY) {
        #pragma omp parallel for schedule(static, 8)
        for (auto &qso : quasars)
            qso->cacheCholeskyCov(p3d_model.get(), CONT_MARG_ENABLED);
        
        preconditioner = [](
                std::unique_ptr<CosmicQuasar> &qso,
                const double *input, double *output
        ) {
            qso->solveCachedCholesky(input, output);
        };
    }
    else {
        fidcosmo::ArinyoP3DModel *p = p3d_model.get();
        preconditioner = [p](
                std::unique_ptr<CosmicQuasar> &qso,
                const double *input, double *output
        ) {
            qso->multInvCov(p, input, output);
        };
    }

    if (CONT_MARG_ENABLED) {
        /* Marginalize. Then, initial guess */
        #pragma omp parallel for schedule(static, 8)
        for (auto &qso : quasars) {
            qso->multInputWithMarg(qso->truth);
            std::swap(qso->truth, qso->in_isig);
            preconditioner(qso, qso->truth, qso->in);
        }
        ioh::continuumMargFileHandler->rewind();
        ++timings["Marg"].first;
        timings["Marg"].second += mytime::timer.getTime() - dt;
    }
    else {
        /* Initial guess */
        #pragma omp parallel for schedule(dynamic, 8)
        for (auto &qso : quasars)
            preconditioner(qso, qso->truth, qso->in);
    }

    multiplyCovVector();

    #pragma omp parallel for reduction(+:init_residual_norm, old_residual_prec, truth_norm)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->residual[i] = qso->truth[i] - qso->out[i];

        // Only search is multiplied until endconjugateGradientDescent
        qso->in = qso->search.get();

        // set search = InvCov . residual
        preconditioner(qso, qso->residual.get(), qso->in);

        init_residual_norm += cblas_ddot(qso->N, qso->residual.get(), 1,
                                         qso->residual.get(), 1);
        old_residual_prec += cblas_ddot(qso->N, qso->residual.get(), 1,
                                        qso->in, 1);
        truth_norm += cblas_ddot(qso->N, qso->truth, 1, qso->truth, 1);
    }

    init_residual_norm = sqrt(init_residual_norm);
    truth_norm = std::max(sqrt(truth_norm), init_residual_norm);
    conv_vec.push_back(init_residual_norm);
    if (absolute_tolerance) truth_norm = 1;

    if (hasConverged(init_residual_norm, truth_norm, tolerance))
        goto endconjugateGradientDescent;

    for (; niter <= max_conj_grad_steps; ++niter) {
        restart = false;
        new_residual_norm = updateY(old_residual_prec);
        restart = new_residual_norm == -1;

        if (restart && verbose)
            LOG::LOGGER.STD("    WARNING: Flat curvature. Restarting.\n");

        if (niter % 100 == 0) {
            if (verbose)  LOG::LOGGER.STD("    Exact calculation of residuals. ");
            for (auto &qso : quasars)
                qso->in = qso->y.get();

            updateYMatrixVectorFunction();
            double true_residual_norm = 0, drift_norm = 0;

            #pragma omp parallel for reduction(+:true_residual_norm, drift_norm)
            for (auto &qso : quasars) {
                for (int i = 0; i < qso->N; ++i) {
                    double r = qso->truth[i] - qso->out[i];
                    qso->out[i] = qso->residual[i] - r;
                    qso->residual[i] = r;
                }

                true_residual_norm += cblas_ddot(
                    qso->N, qso->residual.get(), 1, qso->residual.get(), 1);
                drift_norm += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);
                qso->in = qso->search.get();
            }
            true_residual_norm = sqrt(true_residual_norm);
            drift_norm = sqrt(drift_norm) / true_residual_norm;
            new_residual_norm = true_residual_norm;
            restart |= drift_norm > 0.01;

            if (verbose)
                LOG::LOGGER.STD("Drift: %.2e. %s.\n", drift_norm,
                    restart ? "Restarting" : "Continuing");
        }

        conv_vec.push_back(new_residual_norm / truth_norm);
        double descent_rate = calculateDescentRate();
        bool end_iter = hasConverged(
            new_residual_norm, truth_norm, tolerance, descent_rate);

        if (end_iter)
            goto endconjugateGradientDescent;

        double new_residual_prec = 0;
        // Calculate InvCov . residual into out
        #pragma omp parallel for reduction(+:new_residual_prec)
        for (auto &qso : quasars) {
            // set z (out) = InvCov . residual
            preconditioner(qso, qso->residual.get(), qso->out);
            new_residual_prec += cblas_ddot(qso->N, qso->residual.get(), 1,
                                            qso->out, 1);
        }

        double beta = restart ? 0 : new_residual_prec / old_residual_prec;
        old_residual_prec = new_residual_prec;

        // New direction using preconditioned z = InvCov . residual
        #pragma omp parallel for
        for (auto &qso : quasars) {
            #pragma omp simd
            for (int i = 0; i < qso->N; ++i)
                qso->search[i] = beta * qso->search[i] + qso->out[i];
        }
    }

endconjugateGradientDescent:
    if (verbose)
        LOG::LOGGER.STD(
            "  conjugateGradientDescent finished in %d iterations.\n", niter);

    if (CONT_MARG_ENABLED) {
        double tm1 = mytime::timer.getTime();

        #pragma omp parallel for schedule(static, 8)
        for (auto &qso : quasars) {
            qso->in = qso->y.get();
            qso->multInputWithMarg(qso->in);
            std::copy_n(qso->in_isig, qso->N, qso->in);
            qso->multIsigInVector();
        }
        ioh::continuumMargFileHandler->rewind();
        ++timings["Marg"].first;
        timings["Marg"].second += mytime::timer.getTime() - tm1;
    }
    else {
        #pragma omp parallel for
        for (auto &qso : quasars) {
            qso->in = qso->y.get();
            qso->multIsigInVector();
        }
    }

    ++timings["CGD"].first;
    convergence_file->write(conv_vec.data(), conv_vec.size(),
                            "CGD-" + std::to_string(timings["CGD"].first));
    convergence_file->flush();
    dt = mytime::timer.getTime() - dt;
    timings["CGD"].second += dt;
}


void Qu3DEstimator::multDerivMatrixVec(int i) {
    static size_t
    mesh_kz_max = std::min(
        size_t(ceil(KMAX_EDGE / mesh.k_fund[2])), mesh.ngrid_kz),
    mesh_kz_min = ceil(specifics::MIN_KZ / mesh.k_fund[2]);

    double t1 = mytime::timer.getTime();
    int imu = i / bins::NUMBER_OF_K_BANDS, ik = i % bins::NUMBER_OF_K_BANDS;

    double kmin = std::max(bins::KBAND_CENTERS[ik] - DK_BIN,
                           bins::KBAND_EDGES[0]),
           kmax = std::min(bins::KBAND_CENTERS[ik] + DK_BIN,
                           bins::KBAND_EDGES[bins::NUMBER_OF_K_BANDS]),
           kcenter = bins::KBAND_CENTERS[ik];
    bool is_last_k_bin = ik == (bins::NUMBER_OF_K_BANDS - 1),
         is_first_k_bin = ik == 0;

    std::function<double(double)> legendre_w;
    switch (imu) {
    case 0: legendre_w = legendre0; break;
    case 1: legendre_w = legendre2; break;
    case 2: legendre_w = legendre4; break;
    case 3: legendre_w = legendre6; break;
    default: legendre_w = std::bind(legendre, 2 * imu, std::placeholders::_1);
    }

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        size_t jj = mesh.ngrid_kz * jxy;

        std::fill_n(mesh.field_k.get() + jj, mesh.ngrid_kz, 0);

        double kx, ky;
        double kperp = mesh.getKperpFromIperp(jxy, kx, ky);

        if (fabs(kx) < specifics::MIN_KPERP || fabs(ky) < specifics::MIN_KPERP)
            continue;
        if (kperp >= kmax)
            continue;

        kperp *= kperp;
        for (size_t jz = mesh_kz_min; jz < mesh_kz_max; ++jz) {
            if (jz == 0 && jxy == 0)
                continue;

            double kz = jz * mesh.k_fund[2], kt = sqrt(kz * kz + kperp), alpha, mu;

            if (kt < kmin)  continue;
            else if (kt >= kmax)  break;
            mu = kz / kt;

            if (is_last_k_bin && (kt > kcenter))
                alpha = 1.0;
            else if (is_first_k_bin && (kt < kcenter))
                alpha = 1.0;
            else
                alpha = (1.0 - fabs(kt - kcenter) / DK_BIN);

            alpha *= legendre_w(mu);
            alpha /= kt * kt;
            /* These are handled before in estimateFisherDirect
                     * mesh.invtotalvol
                     * p3d_model->getSpectroWindow2(kz)
                     * mesh.iasgn_window_xy2[jxy]
                     * mesh.iasgn_window_z2[jz]; */

            #ifdef RL_COMP_DERIV
            kt *= radius / rscale_factor;
            alpha *= exp(-kt * kt);
            #endif
            mesh.field_k[jj + jz] = alpha * mesh_rnd.field_k[jj + jz]; 
        }
    }

    mesh.rawFftK2X();

    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->interpMesh2TruthIsig(mesh);

    ++timings["mDerivMatVec"].first;
    timings["mDerivMatVec"].second += mytime::timer.getTime() - t1;
}


void Qu3DEstimator::multiplyDerivVectors(
        double *o1, double *o2, double *lout, const RealField3D *other
) {
    /* Adds current results into o1 (+=). If o2 is nullptr, the operations is
       directly performed on o1. Otherwise, current results first saved into
       a local array, then o1 += lout, and o2 += lout * lout.

       If you pass lout != nullptr, current results are saved into this array.
    */
    static size_t
    mesh_kz_max = std::min(
        size_t(ceil(KMAX_EDGE / mesh.k_fund[2])), mesh.ngrid_kz),
    mesh_kz_min = ceil(specifics::MIN_KZ / mesh.k_fund[2]);
    static auto _lout = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    static auto _spectroWindow2 = [this]() {
        auto ptr = std::make_unique<double[]>(mesh.ngrid_kz);
        for (size_t i = 0; i < mesh.ngrid_kz; ++i) {
            ptr[i] = p3d_model->getSpectroWindow2(i * mesh.k_fund[2]);
        }
        return ptr;
    }();

    double dt = mytime::timer.getTime();

    /* Evolve with Z, save C^-1 . v (in) into mesh  & FFT */
    reverseInterpolateZ(mesh);
    mesh.rawFftX2K();

    if (lout == nullptr)
        lout = (o2 == nullptr) ? o1 : _lout.get();

    std::fill_n(lout, NUMBER_OF_P_BANDS, 0);

    std::function<double(size_t, size_t)> my_norm = RealField3D::getNormFunc(
        &mesh, other, predeconvolve_cic_window);

    #pragma omp parallel for reduction(+:lout[0:NUMBER_OF_P_BANDS]) \
                             schedule(dynamic, 4)
    for (size_t jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        double kx, ky, temp, temp2, temp3;
        double kperp = mesh.getKperpFromIperp(jxy, kx, ky);
        int ik, ik2;

        if (fabs(kx) < specifics::MIN_KPERP || fabs(ky) < specifics::MIN_KPERP)
            continue;
        if (kperp >= KMAX_EDGE)
            continue;

        kperp *= kperp;
        for (size_t jz = mesh_kz_min; jz < mesh_kz_max; ++jz) {
            if (jz == 0 && jxy == 0)
                continue;

            double kz = jz * mesh.k_fund[2], kt = sqrt(kz * kz + kperp), mu;
            if (kt < bins::KBAND_EDGES[0])  continue;
            if (kt >= KMAX_EDGE)  break;

            mu = kz / kt;
            ik = (kt - bins::KBAND_EDGES[0]) / DK_BIN;

            if (kt > bins::KBAND_CENTERS[ik])
                ik2 = std::min(bins::NUMBER_OF_K_BANDS - 1, ik + 1);
            else
                ik2 = std::max(0, ik - 1);

            temp = (1.0 + (jz != 0)) * my_norm(jxy, jz) * _spectroWindow2[jz];
            temp /= kt * kt;
            temp2 = (1.0 - fabs(kt - bins::KBAND_CENTERS[ik]) / DK_BIN);

            #ifdef RL_COMP_DERIV
            kt *= radius / rscale_factor;
            temp *= exp(-kt * kt);
            #endif

            temp3 = temp * (1.0 - temp2);
            temp *= temp2;

            for (int ell = 0; ell < number_of_multipoles; ++ell) {
                temp2 = legendre(2 * ell, mu);
                lout[ik + ell * bins::NUMBER_OF_K_BANDS] += temp2 * temp;
                lout[ik2 + ell * bins::NUMBER_OF_K_BANDS] += temp2 * temp3;
            }
        }
    }

    cblas_dscal(NUMBER_OF_P_BANDS, mesh.invtotalvol, lout, 1);

    if (o2 != nullptr) {
        for (int i = 0; i < NUMBER_OF_P_BANDS; ++i) {
            o1[i] += lout[i];
            o2[i] += lout[i] * lout[i];
        }
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["mDeriv"].first;
    timings["mDeriv"].second += dt;
}


void Qu3DEstimator::estimatePower() {
    LOG::LOGGER.STD("Calculating power spectrum.\n");
    /* calculate Cinv . delta into y */
    conjugateGradientDescent();

    LOG::LOGGER.STD("  Multiplying with derivative matrices.\n");
    /* Evolve with Z, save C^-1 . v (in) into mesh  & FFT */
    multiplyDerivVectors(raw_power.get(), nullptr);

    result_file->write(raw_power.get(), NUMBER_OF_P_BANDS, "FPOWER");
    result_file->flush();
    logTimings();
}


void Qu3DEstimator::filter() {
    if (mympi::this_pe != 0)
        return;

    std::copy_n(fisher.get(), bins::FISHER_SIZE, covariance.get());
    mxhelp::LAPACKE_InvertMatrixLU(covariance.get(), NUMBER_OF_P_BANDS);
    cblas_dgemv(
        CblasRowMajor, CblasNoTrans, NUMBER_OF_P_BANDS, NUMBER_OF_P_BANDS,
        0.5, covariance.get(), NUMBER_OF_P_BANDS,
        raw_power.get(), 1,
        0, filt_power.get(), 1);

    cblas_dgemv(
        CblasRowMajor, CblasNoTrans, NUMBER_OF_P_BANDS, NUMBER_OF_P_BANDS,
        0.5, covariance.get(), NUMBER_OF_P_BANDS,
        raw_bias.get(), 1,
        0, filt_bias.get(), 1);
}


std::string _getFname(std::string x) {
    std::ostringstream buffer(process::FNAME_BASE, std::ostringstream::ate);
    buffer << x << '_' << mympi::this_pe << ".txt";
    return buffer.str();
}


void Qu3DEstimator::write() {
    if (mympi::this_pe != 0)
        return;

    std::string fname = _getFname("_p3d");
    FILE *toWrite = ioh::open_file(fname.c_str(), "w");

    specifics::printBuildSpecifics(toWrite);
    config.writeConfig(toWrite);

    fprintf(toWrite,
        "# -----------------------------------------------------------------\n"
        "# File Template\n# Nk\n"
        "# kperp | kz | P3D | e_P3D | Pfid | d | b | Fd | Fb\n"
        "# Nk     : Number of k bins\n"
        "# k      : k bin [h Mpc^-1]\n"
        "# l      : Multipole\n"
        "# P3D    : Estimated P3D [h^-3 Mpc^3]\n"
        "# e_P3D  : Gaussian error in estimated P3D [h^-3 Mpc^3]\n"
        "# d      : Power estimate before noise (b) subtracted [h^-3 Mpc^3]\n"
        "# b      : Noise estimate [h^-3 Mpc^3]\n"
        "# Fd     : d before Fisher\n"
        "# Fb     : b before Fisher\n"
        "# -----------------------------------------------------------------\n"
    );

    // if (damping_pair.first)
    //     fprintf(toWrite, "# Damped: True\n");
    // else
    //     fprintf(toWrite, "# Damped: False\n");

    // fprintf(toWrite, "# Damping constant: %.3e\n", damping_pair.second);
    fprintf(toWrite, "# %d\n", bins::NUMBER_OF_K_BANDS);
    fprintf(
        toWrite,
        "%14s %s %14s %14s %14s %14s %14s %14s\n", 
        "k", "l", "P3D", "e_P3D", "d", "b", "Fd", "Fb");

    for (int imu = 0; imu < number_of_multipoles; ++imu) {
        for (int ik = 0; ik < bins::NUMBER_OF_K_BANDS; ++ik) {
            size_t i = ik + bins::NUMBER_OF_K_BANDS * imu;
            int l = 2 * imu;
            double k = bins::KBAND_CENTERS[ik],
                   P3D = filt_power[i] - filt_bias[i],
                   e_P3D = sqrt(covariance[i * (NUMBER_OF_P_BANDS + 1)]),
                   d = filt_power[i],
                   b = filt_bias[i],
                   Fd = raw_power[i],
                   Fb = raw_bias[i];
            fprintf(toWrite,
                    "%14e %d %14e %14e %14e %14e %14e %14e\n", 
                    k, l, P3D, e_P3D, d, b, Fd, Fb);
        }
    }

    fclose(toWrite);
    LOG::LOGGER.STD("P3D estimate saved as %s.\n", fname.c_str());

    fname = _getFname("_fisher");
    mxhelp::fprintfMatrix(
        fname.c_str(), fisher.get(),
        NUMBER_OF_P_BANDS, NUMBER_OF_P_BANDS);

    LOG::LOGGER.STD("Fisher matrix saved as %s.\n", fname.c_str());
}


#include "qu3d/optimal_qu3d_mc.cpp"
#include "qu3d/optimal_qu3d_extra.cpp"

int main(int argc, char *argv[]) {
    mympi::init(argc, argv);

    if (argc < 2) {
        fprintf(stderr, "Missing config file!\n");
        return 1;
    }

    const char *FNAME_CONFIG = argv[1];

    myomp::init_fftw();
    gsl_set_error_handler_off();

    ConfigFile config = ConfigFile();

    try {
        config.readFile(FNAME_CONFIG);
        LOG::LOGGER.open(config.get("OutputDir", "."), mympi::this_pe);
        specifics::printBuildSpecifics();
        mytime::writeTimeLogHeader();
    }
    catch (std::exception& e) {
        fprintf(stderr, "Error while reading config file: %s\n", e.what());
        myomp::clean_fftw();
        mympi::finalize();
        return 1;
    }

    try {
        process::readProcess(config);
        bins::readBins(config);
        specifics::readSpecifics(config);
    }
    catch (std::exception& e) {
        LOG::LOGGER.ERR("Error while parsing config file: %s\n",
            e.what());
        myomp::clean_fftw();
        mympi::finalize();
        return 1;
    }

    Qu3DEstimator qps(config);
    bool test_symmetry = config.getInteger("TestSymmetry") > 0;
    bool test_hsqrt = config.getInteger("TestHsqrt") > 0;
    config.checkUnusedKeys();

    if (qps.max_eval_enabled)
        qps.estimateMaxEvals();

    if (test_symmetry)
        qps.testSymmetry();

    if (test_hsqrt)
        qps.testCovSqrt();

    if (qps.test_gaussian_field) {
        if (qps.mock_grid_res_factor > 1) {
            qps.replaceDeltasWithHighResGaussianField();
            goto EndOptimalQu3DNormally;
        }
        else {
            qps.replaceDeltasWithGaussianField();
        }
    }

    qps.estimatePower();

    if (qps.total_bias_enabled)
        qps.estimateTotalBiasMc();

    if (qps.total_bias_direct_enabled)
        qps.estimateTotalBiasDirect();

    if (qps.noise_bias_enabled)
        qps.estimateNoiseBiasMc();

    if (qps.fisher_direct_enabled)
        qps.estimateFisherDirect();

EndOptimalQu3DNormally:
    myomp::clean_fftw();
    mympi::finalize();
    return 0;
}
