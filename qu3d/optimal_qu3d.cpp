#include <unordered_map>

#include "qu3d/optimal_qu3d.hpp"
#include "qu3d/cosmology_3d.hpp"

#include "core/global_numbers.hpp"
#include "core/mpi_manager.hpp"
#include "core/progress.hpp"
#include "io/logger.hpp"

/* Timing map */
std::unordered_map<std::string, std::pair<int, double>> timings{
    {"rInterp", std::make_pair(0, 0.0)}, {"interp", std::make_pair(0, 0.0)},
    {"CGD", std::make_pair(0, 0.0)}, {"mDeriv", std::make_pair(0, 0.0)},
    {"mCov", std::make_pair(0, 0.0)}, {"PPcomp", std::make_pair(0, 0.0)},
    {"GenGauss", std::make_pair(0, 0.0)}
};

/* Internal variables */
std::unordered_map<size_t, std::vector<const CosmicQuasar*>> idx_quasar_map;

std::unique_ptr<fidcosmo::FlatLCDM> cosmo;
std::unique_ptr<fidcosmo::ArinyoP3DModel> p3d_model;

RealField3D mesh_rnd;
std::vector<MyRNG> rngs;
int NUMBER_OF_K_BANDS_2 = 0;
double DK_BIN = 0;
bool verbose = true;
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


void _initRngs(std::seed_seq *seq) {
    const int N = myomp::getMaxNumThreads();
    rngs.resize(N);
    std::vector<size_t> seeds(N);
    seq->generate(seeds.begin(), seeds.end());
    for (int i = 0; i < N; ++i)
        rngs[i].seed(seeds[i]);
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
}


void Qu3DEstimator::_readQSOFiles(
        const std::string &flist, const std::string &findir
) {
    double t1 = mytime::timer.getTime(), t2 = 0;
    std::vector<std::string> filepaths;

    LOG::LOGGER.STD("Read delta files.\n");
    qio::PiccaFile::use_cache = false;

    int number_of_files = ioh::readList(flist.c_str(), filepaths);
    if (mympi::total_pes > 1) {
        int nrot = mympi::this_pe * number_of_files / mympi::total_pes;
        std::rotate(
            filepaths.begin(), filepaths.begin() + nrot,
            filepaths.end());
    }

    #pragma omp parallel for num_threads(8)
    for (auto &fq : filepaths) {
        fq.insert(0, findir);  // Add parent directory to file path
        _readOneDeltaFile(fq);
    }

    t2 = mytime::timer.getTime();

    if (quasars.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");

    int max_qN = 0;
    #pragma omp parallel for reduction(+:num_all_pixels) reduction(max:max_qN)
    for (const auto &qso : quasars) {
        num_all_pixels += qso->N;
        max_qN = std::max(qso->N, max_qN);
    }

    CosmicQuasar::allocCcov(max_qN * max_qN);

    LOG::LOGGER.STD(
        "There are %d quasars and %ld number of pixels. "
        "Reading QSO files took %.2f m.\n",
        quasars.size(), num_all_pixels, t2 - t1);
}


void Qu3DEstimator::_calculateBoxDimensions(double L[3], double &z0) {
    double lxmin = 0, lymin = 0, lzmin = 1e15,
           lxmax = 0, lymax = 0, lzmax = 0;

    #pragma omp parallel for reduction(min:lxmin, lymin, lzmin) \
                             reduction(max:lxmax, lymax, lzmax)
    for (auto it = quasars.cbegin(); it != quasars.cend(); ++it) {
        const CosmicQuasar *qso = it->get();
        lxmin = std::min(lxmin, qso->r[0]);
        lzmin = std::min(lzmin, qso->r[2]);
        lxmax = std::max(lxmax, qso->r[3 * (qso->N - 1)]);
        lzmax = std::max(lzmax, qso->r[3 * qso->N - 1]);

        lymin = std::min(lymin, std::min(qso->r[1], qso->r[3 * qso->N - 2]));
        lymax = std::max(lymax, std::max(qso->r[1], qso->r[3 * qso->N - 2]));
    }

    L[0] = lxmax - lxmin;
    L[1] = lymax - lymin;
    L[2] = lzmax - lzmin;
    z0 = lzmin;
}


void Qu3DEstimator::_setupMesh(double radius) {
    double t1 = mytime::timer.getTime(), t2 = 0;

    _calculateBoxDimensions(mesh.length, mesh.z0);

    mesh.length[1] += 20.0 * radius;
    mesh.length[2] += 20.0 * radius;
    mesh.z0 -= 10.0 * radius;

    mesh.ngrid[0] = config.getInteger("NGRID_X");
    mesh.ngrid[1] = config.getInteger("NGRID_Y");
    mesh.ngrid[2] = config.getInteger("NGRID_Z");

    double dzl = 1.2 * radius - (mesh.length[2] / mesh.ngrid[2]);
    if (dzl > 0) {
        double extra_lz = dzl * mesh.ngrid[2];
        mesh.length[2] += extra_lz;
        mesh.z0 -= extra_lz / 2.0;
    }

    LOG::LOGGER.STD(
        "Box dimensions are as follows: "
        "LX = %.0f Mpc, LY = %.0f Mpc, LZ = %.0f Mpc, Z0: %.0f Mpc.\n",
        mesh.length[0], mesh.length[1], mesh.length[2], mesh.z0);

    mesh.construct(INPLACE_FFT);
    LOG::LOGGER.STD(
        "Mesh cell dimensions are as follows: "
        "dx = %.3f Mpc, dy = %.3f Mpc, dz = %.3f Mpc.\n",
        mesh.dx[0], mesh.dx[1], mesh.dx[2]);

    // Shift coordinates of quasars
    LOG::LOGGER.STD("Shifting quasar locations to center the mesh.\n");
    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->r[1 + 3 * i] += mesh.length[1] / 2;
            qso->r[2 + 3 * i] -= mesh.z0;
        }
        #ifdef COARSE_INTERP
            qso->setCoarseComovingDistances();
        #endif
    }

    LOG::LOGGER.STD("Constructing another mesh for randoms.\n");
    mesh_rnd.copy(mesh);
    mesh_rnd.initRngs(seed_generator.get());
    mesh_rnd.construct(INPLACE_FFT);

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
            return q1->min_x_idx < q2->min_x_idx; }
    );

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


void Qu3DEstimator::_openResultsFile() {
    result_file = std::make_unique<ioh::Qu3dFile>(
        process::FNAME_BASE, mympi::this_pe);

    p3d_model->write(result_file.get());

    double *kperp_grid = raw_power.get(),
           *kz_grid = filt_power.get(),
           *pfid_grid = raw_bias.get();

    for (int iperp = 0; iperp < bins::NUMBER_OF_K_BANDS; ++iperp) {
        for (int iz = 0; iz < bins::NUMBER_OF_K_BANDS; ++iz) {
            size_t i = iz + bins::NUMBER_OF_K_BANDS * iperp;
            kperp_grid[i] = bins::KBAND_CENTERS[iperp];
            kz_grid[i] = bins::KBAND_CENTERS[iz];
            pfid_grid[i] = p3d_model->evaluate(kperp_grid[i], kz_grid[i]);
        }
    }

    result_file->write(kperp_grid, NUMBER_OF_K_BANDS_2, "KPERP");
    result_file->write(kz_grid, NUMBER_OF_K_BANDS_2, "KZ");
    result_file->write(pfid_grid, NUMBER_OF_K_BANDS_2, "PFID");
    result_file->flush();
    std::fill_n(kperp_grid, NUMBER_OF_K_BANDS_2, 0);
    std::fill_n(kz_grid, NUMBER_OF_K_BANDS_2, 0);
    std::fill_n(pfid_grid, NUMBER_OF_K_BANDS_2, 0);
}


Qu3DEstimator::Qu3DEstimator(ConfigFile &configg) : config(configg) {
    config.addDefaults(qu3d_default_parameters);

    num_all_pixels = 0;
    std::string
        flist = config.get("FileNameList"),
        findir = config.get("FileInputDir"),
        seed = config.get("Seed") + std::to_string(mympi::this_pe);

    if (flist.empty())
        throw std::invalid_argument("Must pass FileNameList.");
    if (findir.empty())
        throw std::invalid_argument("Must pass FileInputDir.");

    if (findir.back() != '/')
        findir += '/';

    pp_enabled = config.getInteger("TurnOnPpCovariance") > 0;
    max_conj_grad_steps = config.getInteger("MaxConjGradSteps");
    max_monte_carlos = config.getInteger("MaxMonteCarlos");
    tolerance = config.getDouble("ConvergenceTolerance");
    specifics::DOWNSAMPLE_FACTOR = config.getInteger("DownsampleFactor");
    radius = config.getDouble("LongScale");
    rscale_factor = config.getDouble("ScaleFactor");
    total_bias_enabled = config.getInteger("EstimateTotalBias") > 0;
    noise_bias_enabled = config.getInteger("EstimateNoiseBias") > 0;
    fisher_rnd_enabled = config.getInteger("EstimateFisherFromRandomDerivatives") > 0;

    seed_generator = std::make_unique<std::seed_seq>(seed.begin(), seed.end());
    _initRngs(seed_generator.get());

    cosmo = std::make_unique<fidcosmo::FlatLCDM>(config);
    logCosmoDist(); logCosmoHubble();
    p3d_model = std::make_unique<fidcosmo::ArinyoP3DModel>(config);
    logPmodel();

    _readQSOFiles(flist, findir);
    _setupMesh(radius);
    _constructMap();
    _findNeighbors();

    if (config.getInteger("TestGaussianField") > 0)
        replaceDeltasWithGaussianField();

    radius *= rscale_factor;
    NUMBER_OF_K_BANDS_2 = bins::NUMBER_OF_K_BANDS * bins::NUMBER_OF_K_BANDS;
    DK_BIN = bins::KBAND_CENTERS[1] - bins::KBAND_CENTERS[0];
    bins::FISHER_SIZE = NUMBER_OF_K_BANDS_2 * NUMBER_OF_K_BANDS_2;

    raw_power = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    filt_power = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    raw_bias = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    filt_bias = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    fisher = std::make_unique<double[]>(bins::FISHER_SIZE);
    covariance = std::make_unique<double[]>(bins::FISHER_SIZE);

    _openResultsFile();
}

void Qu3DEstimator::reverseInterpolate() {
    double dt = mytime::timer.getTime();
    mesh.zero_field_x();

    #ifdef COARSE_INTERP
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->coarseGrainIn();

        // Assume 2 threads will not encounter race conditions
        #pragma omp parallel for num_threads(2)
        for (const auto &qso : quasars) {
            for (int i = 0; i < qso->coarse_N; ++i)
                mesh.reverseInterpolateCIC(
                    qso->coarse_r.get() + 3 * i, qso->coarse_in[i]);
        }
    #else
        // Assume 2 threads will not encounter race conditions
        #pragma omp parallel for num_threads(2)
        for (const auto &qso : quasars) {
            for (int i = 0; i < qso->N; ++i)
                mesh.reverseInterpolateCIC(
                    qso->r.get() + 3 * i, qso->in[i]);
        }
    #endif

    dt = mytime::timer.getTime() - dt;
    ++timings["rInterp"].first;
    timings["rInterp"].second += dt;
}


void Qu3DEstimator::reverseInterpolateIsig() {
    double dt = mytime::timer.getTime();
    mesh.zero_field_x();

    #ifdef COARSE_INTERP
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->coarseGrainInIsig();

        // Assume 2 threads will not encounter race conditions
        #pragma omp parallel for num_threads(2)
        for (const auto &qso : quasars) {
            for (int i = 0; i < qso->coarse_N; ++i)
                mesh.reverseInterpolateCIC(
                    qso->coarse_r.get() + 3 * i, qso->coarse_in[i]);
        }
    #else
        // Assume 2 threads will not encounter race conditions
        #pragma omp parallel for num_threads(2)
        for (const auto &qso : quasars) {
            for (int i = 0; i < qso->N; ++i)
                mesh.reverseInterpolateCIC(
                    qso->r.get() + 3 * i, qso->in[i] * qso->isig[i]);
        }
    #endif

    dt = mytime::timer.getTime() - dt;
    ++timings["rInterp"].first;
    timings["rInterp"].second += dt;
}


void Qu3DEstimator::multMeshComp() {
    double t1 = mytime::timer.getTime(), t2 = 0;

    reverseInterpolateIsig();
    // Convolve power. Normalization including cellvol and N^3 yields inverse
    // total volume
    mesh.rawFftX2K();
    #pragma omp parallel for
    for (size_t ij = 0; ij < mesh.ngrid_xy; ++ij) {
        double kperp = mesh.getKperpFromIperp(ij);

        for (int k = 0; k < mesh.ngrid_kz; ++k)
            mesh.field_k[k + mesh.ngrid_kz * ij] *=
                mesh.invtotalvol
                * p3d_model->evaluate(kperp, k * mesh.k_fund[2]);
    }
    mesh.rawFftK2X();

    double dt = mytime::timer.getTime();
    // Interpolate and Weight by isig
    #ifdef COARSE_INTERP
        #pragma omp parallel for
        for (auto &qso : quasars) {
            qso->interpMesh2Coarse(mesh);
            qso->interpNgpCoarse2OutIsig();
        }
    #else
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->interpAddMesh2OutIsig(mesh);
    #endif

    t2 = mytime::timer.getTime();
    ++timings["interp"].first;
    timings["interp"].second += t2 - dt;

    if (verbose)
        LOG::LOGGER.STD("    multMeshComp took %.2f s.\n", 60.0 * (t2 - t1));
}


void Qu3DEstimator::multParticleComp() {
    double t1 = mytime::timer.getTime(), dt = 0;

    #pragma omp parallel for schedule(dynamic, 16)
    for (auto &qso : quasars)
        qso->multCovNeighbors(p3d_model.get());

    dt = mytime::timer.getTime() - t1;
    ++timings["PPcomp"].first;
    timings["PPcomp"].second += dt;

    if (verbose)
        LOG::LOGGER.STD("    multParticleComp took %.2f s.\n", 60.0 * dt);
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

    if (pp_enabled)
        multParticleComp();

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
        LOG::LOGGER.STD("    updateY took %.2f s.\n", 60.0 * t2);
}


void Qu3DEstimator::calculateNewDirection(double beta)  {
    #pragma omp parallel for
    for (auto &qso : quasars) {
        #pragma omp simd
        for (int i = 0; i < qso->N; ++i)
            qso->search[i] = beta * qso->search[i] + qso->residual[i];
    }
}


void Qu3DEstimator::initGuessDiag() {
    double varlss = p3d_model->getVarLss();

    if (verbose)
        LOG::LOGGER.STD("  VarLSS: %.2e.\n", varlss);

    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            double isig = qso->isig[i];
            qso->in[i] = qso->truth[i] / (1.0 + isig * isig * varlss);
        }
    }
}


void Qu3DEstimator::conjugateGradientDescent() {
    double dt = mytime::timer.getTime();
    int niter = 1;

    if (verbose)
        LOG::LOGGER.STD("  Entered conjugateGradientDescent.\n");

    /* Initial guess */
    initGuessDiag();

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

    for (; niter <= max_conj_grad_steps; ++niter) {
        updateY(old_residual_norm2);

        double new_residual_norm2 = calculateResidualNorm2(),
               new_residual_norm = sqrt(new_residual_norm2);

        // bool end_iter = isDiverging(old_residual_norm2, new_residual_norm2);
        bool end_iter = hasConverged(
            new_residual_norm / init_residual_norm, tolerance);

        if (end_iter)
            goto endconjugateGradientDescent;

        double beta = new_residual_norm2 / old_residual_norm2;
        old_residual_norm2 = new_residual_norm2;
        calculateNewDirection(beta);
    }

endconjugateGradientDescent:
    if (verbose)
        LOG::LOGGER.STD(
            "  conjugateGradientDescent finished in %d iterations.\n", niter);

    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->multIsigInVector();

    dt = mytime::timer.getTime() - dt;
    ++timings["CGD"].first;
    timings["CGD"].second += dt;
}


void Qu3DEstimator::multiplyDerivVectors(double *o1, double *o2, double *lout) {
    /* Adds current results into o1 (+=). If o2 is nullptr, the operations is
       directly performed on o1. Otherwise, current results first saved into
       a local array, then o1 += lout, and o2 += lout * lout.

       If you pass lout != nullptr, current results are saved into this array.
    */
    static int mesh_kz_max = std::min(
        int(ceil(bins::KBAND_EDGES[bins::NUMBER_OF_K_BANDS] / mesh.k_fund[2])),
        mesh.ngrid_kz);
    static auto _lout = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);

    double dt = mytime::timer.getTime();

    if (lout == nullptr)
        lout = (o2 == nullptr) ? o1 : _lout.get();

    std::fill_n(lout, NUMBER_OF_K_BANDS_2, 0);

    #pragma omp parallel for reduction(+:lout[0:NUMBER_OF_K_BANDS_2])
    for (size_t jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        int iperp = mesh.getKperpFromIperp(jxy) / DK_BIN;

        if (iperp >= bins::NUMBER_OF_K_BANDS)
            continue;

        size_t jj = mesh.ngrid_kz * jxy;
        lout[iperp * bins::NUMBER_OF_K_BANDS] += std::norm(mesh.field_k[jj]);

        for (int k = 1; k < mesh_kz_max; ++k) {
            int iz = (k * mesh.k_fund[2]) / DK_BIN;
            lout[iz + iperp * bins::NUMBER_OF_K_BANDS] +=
                2.0 * std::norm(mesh.field_k[k + jj]);
        }
    }

    cblas_dscal(NUMBER_OF_K_BANDS_2, mesh.invtotalvol, lout, 1);

    if (o2 != nullptr) {
        for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i) {
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

    reverseInterpolate();
    mesh.rawFftX2K();
    LOG::LOGGER.STD("  Multiplying with derivative matrices.\n");
    multiplyDerivVectors(raw_power.get(), nullptr);

    result_file->write(raw_power.get(), NUMBER_OF_K_BANDS_2, "FPOWER");
    result_file->flush();
    logTimings();
}


bool Qu3DEstimator::_syncMonteCarlo(
        int nmc, double *o1, double *o2, int ndata, const std::string &ext
) {
    bool converged = false;
    std::fill_n(o1, ndata, 0);
    std::fill_n(o2, ndata, 0);
    mympi::reduceToOther(mc1.get(), o1, ndata);
    mympi::reduceToOther(mc2.get(), o2, ndata);

    if (mympi::this_pe == 0) {
        nmc *= mympi::total_pes;
        cblas_dscal(ndata, 1.0 / nmc, o1, 1);
        cblas_dscal(ndata, 1.0 / nmc, o2, 1);
        result_file->write(o1, ndata, ext + "-" + std::to_string(nmc), nmc);
        result_file->write(o2, ndata, ext + "2-" + std::to_string(nmc), nmc);
        result_file->flush();

        double max_std = 0, mean_std = 0, std_k;
        for (int i = 0; i < ndata; ++i) {
            std_k = sqrt(
                (1 - o1[i] * o1[i] / o2[i]) / (nmc - 1)
            );
            max_std = std::max(std_k, max_std);
            mean_std += std_k / ndata;
        }

        LOG::LOGGER.STD(
            "  %d: Estimated relative mean/max std is %.2e/%.2e. "
            "MC converges when < %.2e\n", nmc, mean_std, max_std, tolerance);

        converged = max_std < tolerance;
    }

    mympi::bcast(&converged);
    return converged;
}


void Qu3DEstimator::estimateNoiseBiasMc() {
    LOG::LOGGER.STD("Estimating noise bias.\n");
    verbose = false;
    mc1 = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    mc2 = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);

    Progress prog_tracker(max_monte_carlos, 10);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        /* generate random Gaussian vector into y */
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->fillRngNoise(rngs[myomp::getThreadNum()]);

        /* calculate Cinv . n into y */
        conjugateGradientDescent();

        reverseInterpolate();
        mesh.rawFftX2K();
        multiplyDerivVectors(mc1.get(), mc2.get());

        ++prog_tracker;

        if ((nmc % 5 != 0) && (nmc != max_monte_carlos))
            continue;

        converged = _syncMonteCarlo(
            nmc, raw_bias.get(), filt_bias.get(), NUMBER_OF_K_BANDS_2, "FBIAS");

        if (converged)
            break;
    }

    logTimings();
}


void Qu3DEstimator::estimateTotalBiasMc() {
    /* Saves every Monte Carlo simulation. The results need to be
       post-processed to get the Fisher matrix. */
    LOG::LOGGER.STD("Estimating total bias (S_L + N).\n");
    constexpr int M_MCS = 5;
    verbose = false;
    mc1 = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    mc2 = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);

    // Every task saves their own Monte Carlos
    ioh::Qu3dFile monte_carlos_file(
        process::FNAME_BASE + "-montecarlos-" + std::to_string(mympi::this_pe),
        0);
    auto all_mcs = std::make_unique<double[]>(M_MCS * NUMBER_OF_K_BANDS_2);

    Progress prog_tracker(max_monte_carlos, 10);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        /* generate random Gaussian vector into y */
        replaceDeltasWithGaussianField();

        /* calculate Cinv . n into y */
        conjugateGradientDescent();

        reverseInterpolate();
        mesh.rawFftX2K();
        int jj = (nmc - 1) % M_MCS;
        multiplyDerivVectors(
            mc1.get(), mc2.get(),
            all_mcs.get() + jj * NUMBER_OF_K_BANDS_2
        );

        ++prog_tracker;

        if ((nmc % M_MCS != 0) && (nmc != max_monte_carlos))
            continue;

        monte_carlos_file.write(
            all_mcs.get(), (jj + 1) * NUMBER_OF_K_BANDS_2,
            "TOTBIAS_MCS-" + std::to_string(nmc), jj + 1);

        converged = _syncMonteCarlo(
            nmc, raw_bias.get(), filt_bias.get(), NUMBER_OF_K_BANDS_2,
            "FTOTALBIAS");

        if (converged)
            break;
    }

    logTimings();
}


void Qu3DEstimator::drawRndDeriv(int i) {
    int iperp = i / bins::NUMBER_OF_K_BANDS,
        iz = i % bins::NUMBER_OF_K_BANDS;

    int mesh_z_1 = ceil(bins::KBAND_EDGES[iz] / mesh.k_fund[2]),
        mesh_z_2 = ceil(bins::KBAND_EDGES[iz + 1] / mesh.k_fund[2]);

    mesh_z_1 = std::min(mesh.ngrid_kz, std::max(0, mesh_z_1));
    mesh_z_2 = std::min(mesh.ngrid_kz, mesh_z_2);

    #pragma omp parallel for
    for (size_t jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        size_t jj = mesh.ngrid_kz * jxy;

        std::fill_n(mesh.field_k.begin() + jj, mesh.ngrid_kz, 0);

        double kperp = mesh.getKperpFromIperp(jxy);
        if(isInsideKbin(iperp, kperp))
            for (int zz = mesh_z_1; zz != mesh_z_2; ++zz)
                mesh.field_k[jj + zz] = mesh.invsqrtcellvol * mesh_rnd.field_k[jj + zz];
    }

    mesh.fftK2X();

    #ifdef COARSE_INTERP
        #pragma omp parallel for
        for (auto &qso : quasars) {
            qso->interpMesh2Coarse(mesh);
            qso->interpNgpCoarse2TruthIsig();
        }
    #else
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->interpMesh2TruthIsig(mesh);
    #endif
}


void Qu3DEstimator::estimateFisherFromRndDeriv() {
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
    mc1 = std::make_unique<double[]>(bins::FISHER_SIZE);
    mc2 = std::make_unique<double[]>(bins::FISHER_SIZE);

    /* Create another mesh to store random numbers. This way, randoms are
       generated once, and FFTd once. This grid can perform in-place FFTs,
       since it is only needed in Fourier space, which is equivalent between
       in-place and out-of-place transforms.

       LOG::LOGGER.STD("  Constructing another mesh.\n");
       mesh_rnd.copy(mesh);
       mesh_rnd.initRngs(seed_generator.get());
       mesh_rnd.construct();
    */

    Progress prog_tracker(max_monte_carlos * NUMBER_OF_K_BANDS_2, 5);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        mesh_rnd.fillRndNormal();
        mesh_rnd.fftX2K();
        LOG::LOGGER.STD("  Generated random numbers & FFT.\n");

        for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i) {
            drawRndDeriv(i);

            /* calculate C^-1 . qk into in */
            conjugateGradientDescent();
            /* save C^-1 . v (in) into mesh */
            reverseInterpolate();
            mesh.rawFftX2K();
            multiplyDerivVectors(
                mc1.get() + i * NUMBER_OF_K_BANDS_2,
                mc2.get() + i * NUMBER_OF_K_BANDS_2);
        
            ++prog_tracker;
        }

        converged = _syncMonteCarlo(
            nmc, fisher.get(), covariance.get(), bins::FISHER_SIZE, "2FISHER");

        if (converged)
            break;
    }

    cblas_dscal(bins::FISHER_SIZE, 0.5, fisher.get(), 1);
    logTimings();
}


void Qu3DEstimator::filter() {
    if (mympi::this_pe != 0)
        return;

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

    fprintf(toWrite, "# -----------------------------------------------------------------\n"
        "# File Template\n# Nk\n"
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

    for (int iperp = 0; iperp < bins::NUMBER_OF_K_BANDS; ++iperp) {
        for (int iz = 0; iz < bins::NUMBER_OF_K_BANDS; ++iz) {
            size_t i = iz + bins::NUMBER_OF_K_BANDS * iperp;
            double kperp = bins::KBAND_CENTERS[iperp],
                   kz = bins::KBAND_CENTERS[iz],
                   P3D = filt_power[i] - filt_bias[i],
                   e_P3D = sqrt(covariance[i * (NUMBER_OF_K_BANDS_2 + 1)]),
                   Pfid = p3d_model->evaluate(kperp, kz),
                   d = filt_power[i],
                   b = filt_bias[i],
                   Fd = raw_power[i],
                   Fb = raw_bias[i];
            fprintf(toWrite,
                    "%14e %14e %14e %14e %14e %14e %14e %14e %14e\n", 
                    kperp, kz, P3D, e_P3D, Pfid, d, b, Fd, Fb);
        }
    }

    fclose(toWrite);
    LOG::LOGGER.STD("P3D estimate saved as %s.\n", fname.c_str());

    fname = _getFname("_fisher");
    mxhelp::fprintfMatrix(
        fname.c_str(), fisher.get(),
        NUMBER_OF_K_BANDS_2, NUMBER_OF_K_BANDS_2);

    LOG::LOGGER.STD("Fisher matrix saved as %s.\n", fname.c_str());
}


void Qu3DEstimator::replaceDeltasWithGaussianField() {
    /* Generated deltas will be different between MPI tasks */
    if (verbose)
        LOG::LOGGER.STD("Replacing deltas with Gaussian. ");

    double t1 = mytime::timer.getTime(), t2 = 0;
    /* We could generate a higher resolution grid for truth testing in the
       future. Old code for reference:

       RealField3D mesh_g;
       mesh_g.initRngs(seed_generator.get());
       mesh_g.copy(mesh);
       mesh_g.length[1] *= 1.25; mesh_g.length[2] *= 1.25;
       mesh_g.ngrid[0] *= 2; mesh_g.ngrid[1] *= 2; mesh_g.ngrid[2] *= 2;
       mesh_rnd.construct();
    */

    mesh_rnd.fillRndNormal();
    mesh_rnd.fftX2K();

    #pragma omp parallel for
    for (size_t ij = 0; ij < mesh_rnd.ngrid_xy; ++ij) {
        double kperp = mesh_rnd.getKperpFromIperp(ij);

        for (int k = 0; k < mesh_rnd.ngrid_kz; ++k)
            mesh_rnd.field_k[k + mesh_rnd.ngrid_kz * ij] *=
                mesh_rnd.invsqrtcellvol * sqrt(
                    p3d_model->evaluate(kperp, k * mesh_rnd.k_fund[2])
            );
    }

    mesh_rnd.fftK2X();

    #pragma omp parallel for
    for (auto &qso : quasars) {
        qso->fillRngNoise(rngs[myomp::getThreadNum()]);
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] += qso->isig[i] * mesh_rnd.interpolate(
                qso->r.get() + 3 * i);
    }

    t2 = mytime::timer.getTime() - t1;
    ++timings["GenGauss"].first;
    timings["GenGauss"].second += t2;

    if (verbose)
        LOG::LOGGER.STD("It took %.2f m.\n", t2);
}
