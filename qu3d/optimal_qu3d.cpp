#include <unordered_map>

#include "qu3d/optimal_qu3d.hpp"
#include "qu3d/cosmology_3d.hpp"

#include "core/global_numbers.hpp"
#include "io/logger.hpp"

std::unordered_map<size_t, std::vector<const CosmicQuasar*>> idx_quasar_map;
std::vector<std::pair<size_t, std::vector<const CosmicQuasar*>>> idx_quasars_pairs;

std::unique_ptr<fidcosmo::FlatLCDM> cosmo;
std::unique_ptr<fidcosmo::ArinyoP3DModel> p3d_model;
RealField3D mesh2;
int NUMBER_OF_K_BANDS_2 = 0;


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
}
#else
#define CHECK_ISNAN(X, Y, Z)
void logCosmoDist() {};
void logCosmoHubble() {};
void logPmodel() {};
#endif


inline bool isInsideKbin(int ib, double kb) {
    return (bins::KBAND_EDGES[ib] <= kb) && (kb < bins::KBAND_EDGES[ib + 1]);
}


inline bool hasConverged(double norm, double tolerance) {
    LOG::LOGGER.STD(
        "    Current norm(residuals) is %.8e. "
        "conjugateGradientDescent convergence when < %.2e\n",
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
            std::ostringstream fpath;
            fpath << fname << '[' << i + 1 << ']';
            LOG::LOGGER.ERR(
                "%s. Filename %s.\n", e.what(), fpath.str().c_str());
        }
    }

    pFile.closeFile();

    if (local_quasars.empty())
        return;

    for (auto &qso : local_quasars) {
        qso->setRadialComovingDistance(cosmo.get());
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
    LOG::LOGGER.STD("Reading QSO files took %.2f m.\n", t2 - t1);

    if (quasars.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");


    #pragma omp parallel for reduction(+:num_all_pixels)
    for (auto &qso : quasars)
        num_all_pixels += qso->N;
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

    rscale_long *= rscale_factor;
    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (const size_t &i : qso->grid_indices) {
            auto neighboring_pixels = mesh.findNeighboringPixels(
                i, rscale_long);

            for (const size_t &ipix : neighboring_pixels) {
                auto kumap_itr = idx_quasar_map.find(ipix);

                if (kumap_itr == idx_quasar_map.end())
                    continue;

                qso->neighbors.insert(
                    kumap_itr->second.cbegin(), kumap_itr->second.cend());
            }
        }
    }

    rscale_long /= rscale_factor;
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("_findNeighbors took %.2f m.\n", t2 - t1);
}

Qu3DEstimator::Qu3DEstimator(ConfigFile &config) {
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
    rscale_long = config.getDouble("LongScale");
    rscale_factor = config.getDouble("ScaleFactor");

    mesh.ngrid[0] = config.getInteger("NGRID_X");
    mesh.ngrid[1] = config.getInteger("NGRID_Y");
    mesh.ngrid[2] = config.getInteger("NGRID_Z");

    mesh.length[0] = config.getInteger("LENGTH_X");
    mesh.length[1] = config.getInteger("LENGTH_Y");
    mesh.length[2] = config.getInteger("LENGTH_Z");
    mesh.z0 = config.getInteger("ZSTART");
    mesh2 = mesh;
    double t1 = mytime::timer.getTime(), t2 = 0;
    mesh.construct();
    mesh2.construct();
    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("Mesh construct took %.2f m.\n", t2 - t1);

    NUMBER_OF_K_BANDS_2 = bins::NUMBER_OF_K_BANDS * bins::NUMBER_OF_K_BANDS;
    bins::FISHER_SIZE = NUMBER_OF_K_BANDS_2 * NUMBER_OF_K_BANDS_2;

    power_est = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    bias_est = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);
    fisher = std::make_unique<double[]>(bins::FISHER_SIZE);

    _constructMap();
    // _findNeighbors();
    idx_quasars_pairs.reserve(idx_quasar_map.size());
    for (auto &[idx, qsos] : idx_quasar_map)
        idx_quasars_pairs.push_back(std::make_pair(idx, std::move(qsos)));
    idx_quasar_map.clear();
    rscale_long *= -rscale_long;
}


void Qu3DEstimator::reverseInterpolate() {
    mesh.zero_field_k();

    // double coord[3];
    // for (auto &qso : quasars) {
    //     for (int i = 0; i < qso->N; ++i) {
    //         qso->getCartesianCoords(i, coord);
    //         mesh.reverseInterpolateNGP(coord, qso->in[i]);
    //     }
    // }

    #pragma omp parallel for schedule(dynamic, 1)
    for (const auto &[idx, qsos] : idx_quasars_pairs) {
        double coord[3];
        for (auto &qso : qsos) {
            for (int i = 0; i < qso->N; ++i) {
                qso->getCartesianCoords(i, coord);
                mesh.reverseInterpolateNGP(coord, idx, qso->in[i]);
            }
        }
    }
}


void Qu3DEstimator::multMeshComp() {
    double t1 = mytime::timer.getTime(), t2 = 0;

    reverseInterpolate();

    // Convolve power
    mesh.fftX2K();
    #pragma omp parallel for
    for (size_t i = 0; i < mesh.size_complex; ++i) {
        double k2, kz, p;
        mesh.getK2KzFromIndex(i, k2, kz);
        p = p3d_model->evaluate(sqrt(k2), kz);
        CHECK_ISNAN(&p, 1, "p3d_model");
        mesh.field_k[i] *= p * exp(rscale_long * k2) / mesh.cellvol;
    }
    mesh.fftK2X();

    // Interpolate and Weight by Ivar
    #pragma omp parallel for
    for (auto &qso : quasars) {
        double coord[3];
        for (int i = 0; i < qso->N; ++i) {
            qso->getCartesianCoords(i, coord);
            qso->out[i] += qso->ivar[i] * mesh.interpolate(coord);
        }
    }

    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("    multMeshComp took %.2f m.\n", t2 - t1);
}


double Qu3DEstimator::calculateResidualNorm2() {
    double residual_norm2 = 0;

    #pragma omp parallel for reduction(+:residual_norm2)
    for (const auto &qso : quasars)
        residual_norm2 += cblas_dnrm2(qso->N, qso->residual.get(), 1);

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

    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("    updateY took %.2f m.\n", t2 - t1);
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

    double old_residual_norm2 = calculateResidualNorm2();

    if (hasConverged(sqrt(old_residual_norm2) / num_all_pixels, tolerance))
        return;

    for (int niter = 0; niter < max_conj_grad_steps; ++niter) {
        updateY(old_residual_norm2);

        double new_residual_norm2 = calculateResidualNorm2();

        if (hasConverged(sqrt(new_residual_norm2) / num_all_pixels, tolerance))
            return;

        double beta = new_residual_norm2 / old_residual_norm2;
        old_residual_norm2 = new_residual_norm2;
        calculateNewDirection(beta);
    }
}


void Qu3DEstimator::multiplyDerivVector(int iperp, int iz) {
    int mesh_z_1 = bins::KBAND_EDGES[iz] / mesh.k_fund[2],
        mesh_z_2 = bins::KBAND_EDGES[iz + 1] / mesh.k_fund[2];

    mesh_z_1 = std::max(0, mesh_z_1);
    mesh_z_2 = std::min(mesh.ngrid_kz, mesh_z_2);

    mesh.fftX2K();
    mesh2.zero_field_k();

    #pragma omp parallel for
    for (int jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        double kperp = 0;
        mesh.getKperpFromIperp(jxy, kperp);
        if(!isInsideKbin(iperp, kperp))
            continue;

        auto fk_begin = mesh.field_k.begin() + mesh.ngrid_kz * jxy;
        std::copy(fk_begin + mesh_z_1, fk_begin + mesh_z_2,
                  mesh2.field_k.begin() + mesh.ngrid_kz * jxy);
    }
    mesh.fftK2X();
    mesh2.fftK2X();
}


void Qu3DEstimator::estimatePowerBias() {
    LOG::LOGGER.STD("Calculating power spectrum.\n");
    /* calculate Cinv . delta into y */
    conjugateGradientDescent();

    reverseInterpolate();
    LOG::LOGGER.STD("  Multiplying with derivative matrices.\n");
    for (int iperp = 0; iperp < bins::NUMBER_OF_K_BANDS; ++iperp) {
        for (int iz = 0; iz < bins::NUMBER_OF_K_BANDS; ++iz) {
            /* calculate C,k . y into mesh2 */
            multiplyDerivVector(iperp, iz);

            power_est[iz + bins::NUMBER_OF_K_BANDS * iperp] = mesh.dot(mesh2);
        }
    }

    /* Estimate Bias */
    LOG::LOGGER.STD("Estimating bias. MCs:\n");
    auto old_bias_est = std::make_unique<double[]>(NUMBER_OF_K_BANDS_2);

    for (int nmc = 0; nmc < max_monte_carlos; ++nmc) {
        LOG::LOGGER.STD("%d:", nmc);
        /* generate random Gaussian vector into y */
        #pragma omp parallel for
        for (auto &qso : quasars)
            qso->fillRngNoise();

        /* calculate Cinv . n into y */
        conjugateGradientDescent();

        reverseInterpolate();
        for (int iperp = 0; iperp < bins::NUMBER_OF_K_BANDS; ++iperp) {
            for (int iz = 0; iz < bins::NUMBER_OF_K_BANDS; ++iz) {
                /* calculate C,k . y into mesh2 */
                multiplyDerivVector(iperp, iz);

                old_bias_est[iz + bins::NUMBER_OF_K_BANDS * iperp] = mesh.dot(mesh2);
            }
        }

        double max_rel_err = 0, mean_rel_err = 0;
        for (int i = 0; i < NUMBER_OF_K_BANDS_2; ++i) {
            double rel_err = fabs(bias_est[i] - old_bias_est[i]);
            rel_err /= DOUBLE_EPSILON + std::max(
                fabs(bias_est[i]), fabs(old_bias_est[i]));
            max_rel_err = std::max(rel_err, max_rel_err);
            mean_rel_err += rel_err / NUMBER_OF_K_BANDS_2;
        }

        LOG::LOGGER.STD(
            "  Mean / Max relative errors in bias are %.2e / %.2e. "
            "MC convergences when < %.2e / %.2e\n",
            mean_rel_err, max_rel_err, 1e-6, 1e-4);

        if (mean_rel_err < 1e-6 || max_rel_err < 1e-4)
            break;

        bias_est.swap(old_bias_est);
    }
}

