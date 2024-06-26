#include <unordered_map>

#include "qu3d/optimal_qu3d.hpp"
#include "qu3d/cosmology_3d.hpp"

#include "core/global_numbers.hpp"
#include "core/progress.hpp"
#include "io/logger.hpp"

/* Timing map */
std::unordered_map<std::string, std::pair<int, double>> timings{
    {"rInterp", std::make_pair(0, 0.0)}, {"interp", std::make_pair(0, 0.0)},
    {"CGD", std::make_pair(0, 0.0)}, {"mDeriv", std::make_pair(0, 0.0)},
    {"mCov", std::make_pair(0, 0.0)}
};

/* Internal variables */
std::unordered_map<size_t, std::vector<const CosmicQuasar*>> idx_quasar_map;
std::vector<std::pair<size_t, std::vector<const CosmicQuasar*>>> idx_quasars_pairs;

std::unique_ptr<fidcosmo::FlatLCDM> cosmo;
std::unique_ptr<fidcosmo::ArinyoP3DModel> p3d_model;
RealField3D mesh2;
int NUMBER_OF_K_BANDS_2 = 0;
bool verbose = true;
constexpr bool INPLACE_FFT = false;


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
            " P3d(k=%.2f, kz=%.2f) = %.2f",
            k, kz, p3d_model->evaluate(k, kz));
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


inline bool isInsideKbin(int ib, double kb) {
    return (bins::KBAND_EDGES[ib] <= kb) && (kb < bins::KBAND_EDGES[ib + 1]);
}


inline bool isDiverging(double old_norm, double new_norm) {
    double a = std::max(old_norm, new_norm);
    bool diverging = (old_norm - new_norm) < DOUBLE_EPSILON * a;
    if (verbose && diverging)
        LOG::LOGGER.STD("    Iterations are stagnant or diverging.\n");

    return diverging;
}


inline bool hasConverged(double norm, double tolerance) {
    if (verbose)
        LOG::LOGGER.STD(
            "    Current norm(residuals) / norm(initial residuals) is %.8e. "
            "conjugateGradientDescent converges when this is < %.2e\n",
            norm, tolerance);

    return norm < tolerance;
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

    for (auto &qso : local_quasars) {
        qso->setComovingDistances(cosmo.get());
        CHECK_ISNAN(qso->r.get(), qso->N, "comovingdist");
    }

    #pragma omp critical
    {
        quasars.reserve(quasars.size() + local_quasars.size());
        std::move(std::begin(local_quasars), std::end(local_quasars),
                  std::back_inserter(quasars));
    }

    local_quasars.clear();
}


void Qu3DEstimator::_readQSOFiles(
        const std::string &flist, const std::string &findir
) {
    double t1 = mytime::timer.getTime(), t2 = 0;
    std::vector<std::string> filepaths;

    LOG::LOGGER.STD("Read delta files.\n");
    qio::PiccaFile::use_cache = false;

    int number_of_files = ioh::readList(flist.c_str(), filepaths);

    #pragma omp parallel for num_threads(8)
    for (auto &fq : filepaths) {
        fq.insert(0, findir);  // Add parent directory to file path
        _readOneDeltaFile(fq);
    }

    t2 = mytime::timer.getTime();

    if (quasars.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");


    #pragma omp parallel for reduction(+:num_all_pixels)
    for (auto &qso : quasars)
        num_all_pixels += qso->N;

    LOG::LOGGER.STD(
        "There are %d quasars and %ld number of pixels. "
        "Reading QSO files took %.2f m.\n",
        quasars.size(), num_all_pixels, t2 - t1);
}


void Qu3DEstimator::_constructMap() {
    double t1 = mytime::timer.getTime(), t2 = 0;

    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->findGridPoints(mesh);

    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("findGridPoints took %.2f m.\n", t2 - t1);

    for (const auto &qso : quasars)
        for (const auto &i : qso->grid_indices)
            idx_quasar_map[i].push_back(qso.get());

    t1 = mytime::timer.getTime();
    LOG::LOGGER.STD("Appending map took %.2f m.\n", t1 - t2);
}


void Qu3DEstimator::_findNeighbors() {
    double t1 = mytime::timer.getTime(), t2 = 0;

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
    }

    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("_findNeighbors took %.2f m.\n", t2 - t1);
}

Qu3DEstimator::Qu3DEstimator(ConfigFile &configg) : config(configg) {
    config.addDefaults(qu3d_default_parameters);

    num_all_pixels = 0;
    std::string
        flist = config.get("FileNameList"),
        findir = config.get("FileInputDir");

    if (flist.empty())
        throw std::invalid_argument("Must pass FileNameList.");
    if (findir.empty())
        throw std::invalid_argument("Must pass FileInputDir.");

    if (findir.back() != '/')
        findir += '/';

    cosmo = std::make_unique<fidcosmo::FlatLCDM>(config);
    logCosmoDist(); logCosmoHubble();
    p3d_model = std::make_unique<fidcosmo::ArinyoP3DModel>(config);
    logPmodel();

    _readQSOFiles(flist, findir);

    max_conj_grad_steps = config.getInteger("MaxConjGradSteps");
    max_monte_carlos = config.getInteger("MaxMonteCarlos");
    tolerance = config.getDouble("ConvergenceTolerance");
    specifics::DOWNSAMPLE_FACTOR = config.getInteger("DownsampleFactor");
    radius = config.getDouble("LongScale");
    rscale_factor = config.getDouble("ScaleFactor");
    radius *= rscale_factor;

    mesh.ngrid[0] = config.getInteger("NGRID_X");
    mesh.ngrid[1] = config.getInteger("NGRID_Y");
    mesh.ngrid[2] = config.getInteger("NGRID_Z");

    mesh.length[0] = config.getInteger("LENGTH_X");
    mesh.length[1] = config.getInteger("LENGTH_Y");
    mesh.length[2] = config.getInteger("LENGTH_Z");
    mesh.z0 = config.getInteger("ZSTART");
    mesh2.copy(mesh);
    double t1 = mytime::timer.getTime(), t2 = 0;
    mesh.construct(INPLACE_FFT);
    mesh2.construct(INPLACE_FFT);
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("Mesh construct took %.2f m.\n", t2 - t1);

    NUMBER_OF_K_BANDS_2 = bins::NUMBER_OF_K_BANDS * bins::NUMBER_OF_K_BANDS;
    bins::FISHER_SIZE = NUMBER_OF_K_BANDS_2 * NUMBER_OF_K_BANDS_2;

    raw_power = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    filt_power = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    raw_bias = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    filt_bias = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    fisher = std::make_unique<double[]>(bins::FISHER_SIZE);
    covariance = std::make_unique<double[]>(bins::FISHER_SIZE);

    _constructMap();
    // _findNeighbors();
    // idx_quasars_pairs.reserve(idx_quasar_map.size());
    // for (auto &[idx, qsos] : idx_quasar_map)
    //     idx_quasars_pairs.push_back(std::make_pair(idx, std::move(qsos)));
    // std::sort(idx_quasars_pairs.begin(), idx_quasars_pairs.end());
    idx_quasar_map.clear();
}

void Qu3DEstimator::reverseInterpolate() {
    double dt = mytime::timer.getTime();
    mesh.zero_field_x();

    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            mesh.reverseInterpolateCIC(qso->r.get() + 3 * i, qso->in[i]);

    dt = mytime::timer.getTime() - dt;
    ++timings["rInterp"].first;
    timings["rInterp"].second += dt;
}


void Qu3DEstimator::reverseInterpolateIsig() {
    double dt = mytime::timer.getTime();
    mesh.zero_field_x();

    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            mesh.reverseInterpolateCIC(
                qso->r.get() + 3 * i, qso->in[i] * qso->isig[i]);

    dt = mytime::timer.getTime() - dt;
    ++timings["rInterp"].first;
    timings["rInterp"].second += dt;
    // Not faster
    // #pragma omp parallel for schedule(dynamic, 8) num_threads(8)
    // for (const auto &[idx, qsos] : idx_quasars_pairs) {
    //     double coord[3];
    //     for (const auto &qso : qsos) {
    //         for (int i = 0; i < qso->N; ++i) {
    //             qso->getCartesianCoords(i, coord);
    //             mesh.reverseInterpolateNGP(coord, idx, qso->in[i]);
    //         }
    //     }
    // }
}


void Qu3DEstimator::multMeshComp() {
    double t1 = mytime::timer.getTime(), t2 = 0;

    reverseInterpolateIsig();

    // Convolve power
    mesh.fftX2K();
    #pragma omp parallel for
    for (size_t i = 0; i < mesh.size_complex; ++i) {
        double k2, kz;
        mesh.getK2KzFromIndex(i, k2, kz);
        mesh.field_k[i] *= p3d_model->evaluate(sqrt(k2), kz) / mesh.cellvol;
    }
    mesh.fftK2X();

    double dt = mytime::timer.getTime();
    // Interpolate and Weight by isig
    #pragma omp parallel for
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->out[i] += qso->isig[i] * mesh.interpolate(qso->r.get() + 3 * i);

    t2 = mytime::timer.getTime();
    ++timings["interp"].first;
    timings["interp"].second += t2 - dt;

    if (verbose)
        LOG::LOGGER.STD("    multMeshComp took %.2f m.\n", t2 - t1);
}


void Qu3DEstimator::multiplyCovVector() {
    /* Multiply each quasar's *in pointer and save to *out pointer.
       (I + N^-1/2 S N^-1/2) z = out
    */
    double dt = mytime::timer.getTime();

    // init new results to Cy = I.y
    #pragma omp parallel for
    for (auto &qso : quasars)
        std::copy_n(qso->in, qso->N, qso->out);

    // Add long wavelength mode to Cy
    multMeshComp();
    // multParticleComp();
    dt = mytime::timer.getTime() - dt;
    ++timings["mCov"].first;
    timings["mCov"].second += dt;
}


double Qu3DEstimator::calculateResidualNorm2() {
    double residual_norm2 = 0;

    #pragma omp parallel for reduction(+:residual_norm2)
    for (const auto &qso : quasars) {
        double rn = cblas_dnrm2(qso->N, qso->residual.get(), 1);
        residual_norm2 += rn * rn;
    }

    return residual_norm2;
}


void Qu3DEstimator::updateY(double residual_norm2) {
    double t1 = mytime::timer.getTime(), t2 = 0;

    double a_down = 0, alpha = 0;

    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->in = qso->search.get();

    /* Multiply C x search into Cy*/
    multiplyCovVector();

    // get a_down
    #pragma omp parallel for reduction(+:a_down)
    for (const auto &qso : quasars)
        a_down += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);

    alpha = residual_norm2 / a_down;

    /* Update y in the search direction, restore qso->in
       Update residual */
    #pragma omp parallel for
    for (auto &qso : quasars) {
        /* in is search.get() */
        cblas_daxpy(qso->N, alpha, qso->in, 1, qso->y.get(), 1);
        cblas_daxpy(qso->N, -alpha, qso->out, 1, qso->residual.get(), 1);
        qso->in = qso->y.get();
    }

    t2 = mytime::timer.getTime() - t1;
    if (verbose)
        LOG::LOGGER.STD("    updateY took %.2f m.\n", t2);
}


void Qu3DEstimator::calculateNewDirection(double beta)  {
    #pragma omp parallel for
    for (auto &qso : quasars) {
        // cblas_dscal(qso->N, beta, qso->search.get(), 1);
        // cblas_daxpy(qso->N, 1, qso->residual.get(), 1, qso->search.get(), 1);
        for (int i = 0; i < qso->N; ++i)
            qso->search[i] = beta * qso->search[i] + qso->residual[i];
    }
}


void Qu3DEstimator::conjugateGradientDescent() {
    double dt = mytime::timer.getTime();
    if (verbose)
        LOG::LOGGER.STD("  Entered conjugateGradientDescent.\n");

    /* Initial guess */
    #pragma omp parallel for
    for (auto &qso : quasars)
        std::copy_n(qso->truth, qso->N, qso->in);

    multiplyCovVector();

    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->residual[i] = qso->truth[i] - qso->out[i];
            qso->search[i] = qso->residual[i];
        }
    }

    double old_residual_norm2 = calculateResidualNorm2(),
           init_residual_norm = sqrt(old_residual_norm2);

    if (hasConverged(init_residual_norm, tolerance))
        goto endconjugateGradientDescent;

    for (int niter = 0; niter < max_conj_grad_steps; ++niter) {
        updateY(old_residual_norm2);

        double new_residual_norm2 = calculateResidualNorm2(),
               new_residual_norm = sqrt(new_residual_norm2);

        bool end_iter = isDiverging(old_residual_norm2, new_residual_norm2);
        end_iter |= hasConverged(new_residual_norm / init_residual_norm, tolerance);

        if (end_iter)
            goto endconjugateGradientDescent;

        double beta = new_residual_norm2 / old_residual_norm2;
        old_residual_norm2 = new_residual_norm2;
        calculateNewDirection(beta);
    }

endconjugateGradientDescent:
    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->multIsigInVector();

    dt = mytime::timer.getTime() - dt;
    ++timings["CGD"].first;
    timings["CGD"].second += dt;
}


double Qu3DEstimator::multiplyDerivVector(int i) {
    double dt = mytime::timer.getTime();
    int iperp = i / bins::NUMBER_OF_K_BANDS,
        iz = i % bins::NUMBER_OF_K_BANDS;

    int mesh_z_1 = bins::KBAND_EDGES[iz] / mesh.k_fund[2],
        mesh_z_2 = bins::KBAND_EDGES[iz + 1] / mesh.k_fund[2];

    mesh_z_1 = std::min(mesh.ngrid_kz, std::max(0, mesh_z_1));
    mesh_z_2 = std::min(mesh.ngrid_kz, mesh_z_2);

    #pragma omp parallel for
    for (int jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        size_t jj = mesh.ngrid_kz * jxy;

        double kperp = mesh.getKperpFromIperp(jxy);

        if(!isInsideKbin(iperp, kperp)) {
            std::fill(
                mesh2.field_k.begin() + jj,
                mesh2.field_k.begin() + jj + mesh.ngrid_kz, 0);
        }
        else {
            std::fill(
                mesh2.field_k.begin() + jj,
                mesh2.field_k.begin() + jj + mesh_z_1, 0);
            std::copy(
                mesh.field_k.begin() + jj + mesh_z_1,
                mesh.field_k.begin() + jj + mesh_z_2,
                mesh2.field_k.begin() + jj + mesh_z_1);
            std::fill(
                mesh2.field_k.begin() + jj + mesh_z_2,
                mesh2.field_k.begin() + jj + mesh.ngrid_kz, 0);
        }
    }
    mesh2.fftK2X();

    double result = mesh.dot(mesh2);
    dt = mytime::timer.getTime() - dt;
    ++timings["mDeriv"].first;
    timings["mDeriv"].second += dt;
    return result;
}


void Qu3DEstimator::estimatePower() {
    LOG::LOGGER.STD("Calculating power spectrum.\n");
    /* calculate Cinv . delta into y */
    conjugateGradientDescent();

    reverseInterpolate();
    mesh.fftX2K();
    LOG::LOGGER.STD("  Multiplying with derivative matrices.\n");
    /* calculate C,k . y into mesh2, then dot */
    for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i)
        raw_power[i] = multiplyDerivVector(i) / mesh.cellvol;

    logTimings();
}


void Qu3DEstimator::estimateBiasMc() {
    LOG::LOGGER.STD("Estimating bias.\n");
    verbose = false;
    auto total_b = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    auto total_b2 = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);

    Progress prog_tracker(max_monte_carlos, 10);
    for (int nmc = 1; nmc <= max_monte_carlos; ++nmc) {
        /* generate random Gaussian vector into y */
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->fillRngNoise();

        /* calculate Cinv . n into y */
        conjugateGradientDescent();

        reverseInterpolate();
        mesh.fftX2K();
        for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i) {
            /* calculate C,k . y into mesh2 */
            double b = multiplyDerivVector(i) / mesh.cellvol;
            total_b[i] += b;
            total_b2[i] += b * b;
        }

        ++prog_tracker;

        for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i)
            raw_bias[i] = total_b[i] / nmc;

        if ((nmc < 20) || (nmc % 5 != 0))
            continue;

        double max_std = 0, mean_std = 0, std_k, b2_k;
        for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i) {
            b2_k = total_b2[i] / nmc;
            std_k = sqrt(
                (1 - raw_bias[i] * raw_bias[i] / b2_k) / (nmc - 1)
            );
            max_std = std::max(std_k, max_std);
            mean_std += std_k / NUMBER_OF_K_BANDS_2;
        }

        LOG::LOGGER.STD(
            "  %d: Estimated relative mean/max std is %.2e/%.2e. "
            "MC converges when < %.2e\n", nmc, mean_std, max_std, tolerance);

        if (max_std < tolerance)
            break;
    }

    logTimings();
}


void Qu3DEstimator::drawRndDeriv(int i) {
    int iperp = i / bins::NUMBER_OF_K_BANDS,
        iz = i % bins::NUMBER_OF_K_BANDS;

    int mesh_z_1 = bins::KBAND_EDGES[iz] / mesh.k_fund[2],
        mesh_z_2 = bins::KBAND_EDGES[iz + 1] / mesh.k_fund[2];

    mesh_z_1 = std::min(mesh.ngrid_kz, std::max(0, mesh_z_1));
    mesh_z_2 = std::min(mesh.ngrid_kz, mesh_z_2);

    mesh.fillRndNormal();
    mesh.fftX2K();

    #pragma omp parallel for
    for (int jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        size_t jj = mesh.ngrid_kz * jxy;

        double kperp = mesh.getKperpFromIperp(jxy);

        if(!isInsideKbin(iperp, kperp)) {
            std::fill(
                mesh.field_k.begin() + jj,
                mesh.field_k.begin() + jj + mesh.ngrid_kz, 0);
        }
        else {
            std::fill(
                mesh.field_k.begin() + jj,
                mesh.field_k.begin() + jj + mesh_z_1, 0);
            std::fill(
                mesh.field_k.begin() + jj + mesh_z_2,
                mesh.field_k.begin() + jj + mesh.ngrid_kz, 0);
        }
    }

    mesh.fftK2X();

    #pragma omp parallel for
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] = qso->isig[i] * mesh.interpolate(qso->r.get() + 3 * i);
}


void Qu3DEstimator::estimateFisher() {
    /* Tr[C^-1 . Qk . C^-1 Qk'] = <qk^T . C^-1 . Qk' . C^-1 . qk>

    1. Generate qk on the grid.
        - Generate uniform white noise on x. FFT.
        - Cut out k modes. iFFT.
    2. Interpolate . isig to each qso->truth
    3. Solve C^-1 . qk into qso->in
    4. Reverse interpolate to mesh.
    5. Multiply with Qk' on mesh.
    */
    LOG::LOGGER.STD("Estimating Fisher.\n");
    verbose = false;

    Progress prog_tracker(max_monte_carlos);
    for (int nmc = 1; nmc <= max_monte_carlos; ++nmc) {
        for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i) {
            drawRndDeriv(i);

            /* calculate C^-1 . qk into in */
            conjugateGradientDescent();
            /* save C^-1 . v (in) into mesh */
            reverseInterpolate();
            mesh.fftX2K();

            /* calculate C,k . y into mesh2 */
            for (int j = i; j < NUMBER_OF_K_BANDS_2; ++j)
                fisher[j + i * NUMBER_OF_K_BANDS_2] += multiplyDerivVector(j);
        }
        ++prog_tracker;
    }

    double alpha = 0.5 / (max_monte_carlos * mesh.cellvol);
    cblas_dscal(bins::FISHER_SIZE, alpha, fisher.get(), 1);
    mxhelp::copyUpperToLower(fisher.get(), NUMBER_OF_K_BANDS_2);
}


void Qu3DEstimator::filter() {
    std::copy_n(fisher.get(), bins::FISHER_SIZE, covariance.get());
    mxhelp::LAPACKE_InvertMatrixLU(covariance.get(), NUMBER_OF_K_BANDS_2);
    cblas_dgemv(
        CblasRowMajor, CblasNoTrans, NUMBER_OF_K_BANDS_2, NUMBER_OF_K_BANDS_2,
        0.5, covariance.get(), NUMBER_OF_K_BANDS_2,
        raw_power.get(), 1,
        0, filt_power.get(), 1);

    cblas_dgemv(
        CblasRowMajor, CblasNoTrans, NUMBER_OF_K_BANDS_2, NUMBER_OF_K_BANDS_2,
        0.5, covariance.get(), NUMBER_OF_K_BANDS_2,
        raw_bias.get(), 1,
        0, filt_bias.get(), 1);
}


void Qu3DEstimator::write() {
    std::string buffer = process::FNAME_BASE + "_p3d.txt";

    FILE *toWrite = ioh::open_file(buffer.c_str(), "w");

    specifics::printBuildSpecifics(toWrite);
    config.writeConfig(toWrite);

    fprintf(toWrite, "# -----------------------------------------------------------------\n"
        "# File Template\nNk\n"
        "# kperp | kz | P3D | e_P3D | Pfid | d | b | Fd | Fb\n"
        "# Nk     : Number of k bins\n"
        "# kperp  : Perpendicular k bin [Mpc^-1]\n"
        "# kz     : Line-of-sight k bin [Mpc^-1]\n"
        "# P3D    : Estimated P3D [Mpc^3]\n"
        "# e_P3D  : Gaussian error in estimated P3D [Mpc^3]\n"
        "# Pfid   : Fiducial power [Mpc^3]\n"
        "# d      : Power estimate before noise (b) subtracted [Mpc^3]\n"
        "# b      : Noise estimate [Mpc^3]\n"
        "# Fd     : d before Fisher\n"
        "# Fb     : b before Fisher\n"
        "# -----------------------------------------------------------------\n");

    // if (damping_pair.first)
    //     fprintf(toWrite, "# Damped: True\n");
    // else
    //     fprintf(toWrite, "# Damped: False\n");

    // fprintf(toWrite, "# Damping constant: %.3e\n", damping_pair.second);
    fprintf(toWrite, "# %d\n", bins::NUMBER_OF_K_BANDS);
    fprintf(
        toWrite,
        "%14s %14s %14s %14s %14s %14s %14s %14s %14s\n", 
        "kperp", "kz", "P3D", "e_P3D", "Pfid", "d", "b", "Fd", "Fb");

    int kn = 0, zm = 0;
    for (int iperp = 0; iperp < bins::NUMBER_OF_K_BANDS; ++iperp) {
        for (int iz = 0; iz < bins::NUMBER_OF_K_BANDS; ++iz) {
            size_t i = iz + bins::NUMBER_OF_K_BANDS * iperp;
            double kperp = bins::KBAND_CENTERS[kperp],
                   kz = bins::KBAND_CENTERS[iz],
                   P3D = filt_power[i] - filt_bias[i],
                   e_P3D = sqrt(covariance[i * (NUMBER_OF_K_BANDS_2 + 1)]),
                   k = sqrt(kperp * kperp + kz * kz),
                   Pfid = p3d_model->evaluate(k, kz),
                   d = filt_power[i],
                   b = filt_bias[i],
                   Fd = raw_power[iz + bins::NUMBER_OF_K_BANDS * iperp],
                   Fb = raw_bias[iz + bins::NUMBER_OF_K_BANDS * iperp];
            fprintf(toWrite,
                    "%14e %14e %14e %14e %14e %14e %14e %14e %14e\n", 
                    kperp, kz, P3D, e_P3D, Pfid, d, b, Fd, Fb);
        }
    }

    fclose(toWrite);
    LOG::LOGGER.STD("P3D estimate saved as %s.\n", buffer.c_str());

    buffer = process::FNAME_BASE + "_fisher.txt";
    mxhelp::fprintfMatrix(
        buffer.c_str(), fisher.get(), NUMBER_OF_K_BANDS_2, NUMBER_OF_K_BANDS_2);

    LOG::LOGGER.STD("Fisher matrix saved as %s.\n", buffer.c_str());
}
