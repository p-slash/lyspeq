void Qu3DEstimator::multiplyIpHVector(double m) {
    /* m I + I + N^-1/2 G^1/2 (S_L + S_OD) G^1/2 N^-1/2 A_BD^-1
        input is const *in, output is *out
        uses: *in_isig
    */
    double dt = mytime::timer.getTime();

    /* A_BD^-1. Might not be true if pp_enabled=false */
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        qso->multInvCov(p3d_model.get(), qso->in, qso->out, pp_enabled);
        double *tmp_in = qso->in;
        qso->in = qso->out;
        qso->setInIsigNoMarg();
        qso->in = tmp_in;
    }

    // Add long wavelength mode to Cy
    // only in_isig is used until (B)
    multMeshComp();

    if (pp_enabled) {
        double t1_pp = mytime::timer.getTime(), dt_pp = 0;

        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars)
            if (!qso->neighbors.empty())
                qso->multCovNeighborsOnly(p3d_model.get(), effective_chi);

        dt_pp = mytime::timer.getTime() - t1_pp;
        ++timings["PPcomp"].first;
        timings["PPcomp"].second += dt_pp;

        if (verbose)
            LOG::LOGGER.STD("    multParticleComp took %.2f s.\n", 60.0 * dt_pp);
    }
    // (B)

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->out[i] *= qso->isig[i] * qso->z1[i];
            qso->out[i] += (1.0 + m) * qso->in[i];
        }
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["mIpH"].first;
    timings["mIpH"].second += dt;
}


void Qu3DEstimator::conjugateGradientIpH() {
    double dt = mytime::timer.getTime();
    int niter = 1;

    double init_residual_norm = 0, old_residual_prec = 0,
           new_residual_norm = 0;

    updateYMatrixVectorFunction = [this]() { this->multiplyIpHVector(1.0); };

    const double precon_diag = 2.0 + (
        1.0 - p3d_model->getVar1dS() / p3d_model->getVar1dT());

    if (verbose)
        LOG::LOGGER.STD("  Entered conjugateGradientIpH. Preconditioner %.5f\n",
                        precon_diag);

    /* Initial guess */
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->in[i] = qso->truth[i] / precon_diag;

    multiplyIpHVector(1.0);

    #pragma omp parallel for schedule(dynamic, 4) \
                             reduction(+:init_residual_norm, old_residual_prec)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->residual[i] = qso->truth[i] - qso->out[i];

        // Only search is multiplied from here until endconjugateGradientIpH
        qso->in = qso->search.get();

        // set search = PreCon . residual
        for (int i = 0; i < qso->N; ++i)
            qso->in[i] = qso->residual[i] / precon_diag;

        init_residual_norm += cblas_ddot(
            qso->N, qso->residual.get(), 1, qso->residual.get(), 1);
        old_residual_prec += cblas_ddot(qso->N, qso->residual.get(), 1, qso->in, 1);
    }

    init_residual_norm = sqrt(init_residual_norm);

    if (hasConverged(init_residual_norm, tolerance))
        goto endconjugateGradientIpH;

    if (absolute_tolerance) init_residual_norm = 1;

    for (; niter <= max_conj_grad_steps; ++niter) {
        new_residual_norm = updateY(old_residual_prec);

        bool end_iter = hasConverged(
            new_residual_norm / init_residual_norm, tolerance);

        if (end_iter)
            goto endconjugateGradientIpH;

        double new_residual_prec = 0;
        // Calculate PreCon . residual into out
        #pragma omp parallel for schedule(dynamic, 4) reduction(+:new_residual_prec)
        for (auto &qso : quasars) {
            // set z (out) = PreCon . residual
            for (int i = 0; i < qso->N; ++i)
                qso->out[i] = qso->residual[i] / precon_diag;

            new_residual_prec += cblas_ddot(qso->N, qso->residual.get(), 1, qso->out, 1);
        }

        double beta = new_residual_prec / old_residual_prec;
        old_residual_prec = new_residual_prec;

        // New direction using preconditioned z = Precon . residual
        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars) {
            #pragma omp simd
            for (int i = 0; i < qso->N; ++i)
                qso->search[i] = beta * qso->search[i] + qso->out[i];
        }
    }

endconjugateGradientIpH:
    if (verbose)
        LOG::LOGGER.STD(
            "  endconjugateGradientIpH finished in %d iterations.\n", niter);

    for (auto &qso : quasars)
        qso->in = qso->y.get();

    dt = mytime::timer.getTime() - dt;
    ++timings["cgdIpH"].first;
    timings["cgdIpH"].second += dt;
}


#if 1
void Qu3DEstimator::multiplyHsqrt() {
    /* multiply with SquareRootMatrix: 0.25 + H (0.25 + (I+H)^-1)
        input is *truth, output is *truth
        uses: *in, *in_isig, *sc_eta, *out
    */

    // (1) CG solve I + H from *truth to *in
    //     add 0.25 *truth to *in
    conjugateGradientIpH();

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->in[i] += 0.25 * qso->truth[i];
    }

    // (2) multiply *in with H save to *out
    multiplyIpHVector(0.0);

    // (3) add 0.25 *truth to *out
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] = 0.25 * qso->truth[i] + qso->out[i];
}
#else
void Qu3DEstimator::multiplyHsqrt() {
    for (auto &qso : quasars)
        std::swap(qso->truth, qso->in);

    multiplyIpHVector(1.0);

    for (auto &qso : quasars) {
        std::swap(qso->truth, qso->in);
        std::swap(qso->truth, qso->out);

        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] /= 2.0;
    }
}
#endif


void Qu3DEstimator::replaceDeltasWithGaussianField() {
    if (verbose)
        LOG::LOGGER.STD("Replacing deltas with Gaussian. ");

    double t1 = mytime::timer.getTime(), t2 = 0;
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        qso->blockRandom(rngs[myomp::getThreadNum()], p3d_model.get());

    multiplyHsqrt();

    t2 = mytime::timer.getTime() - t1;
    ++timings["GenGauss"].first;
    timings["GenGauss"].second += t2;

    if (verbose)
        LOG::LOGGER.STD("It took %.2f m.\n", t2);
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


void Qu3DEstimator::estimateTotalBiasMc() {
    /* Saves every Monte Carlo simulation. The results need to be
       post-processed to get the Fisher matrix. */
    LOG::LOGGER.STD("Estimating total bias (S_L + N).\n");
    constexpr int M_MCS = 5;
    verbose = false;
    mc1 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    mc2 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);

    // Every task saves their own Monte Carlos
    ioh::Qu3dFile monte_carlos_file(
        process::FNAME_BASE + "-montecarlos-" + std::to_string(mympi::this_pe),
        0);
    auto all_mcs = std::make_unique<double[]>(M_MCS * NUMBER_OF_P_BANDS);

    Progress prog_tracker(max_monte_carlos, 10);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        /* generate random Gaussian vector into truth */
        replaceDeltasWithGaussianField();

        /* calculate Cinv . n into y */
        conjugateGradientDescent();

        int jj = (nmc - 1) % M_MCS;
        /* Evolve with Z, save C^-1 . v (in) into mesh  & FFT */
        multiplyDerivVectors(
            mc1.get(), mc2.get(),
            all_mcs.get() + jj * NUMBER_OF_P_BANDS
        );

        ++prog_tracker;

        if ((nmc % M_MCS != 0) && (nmc != max_monte_carlos))
            continue;

        monte_carlos_file.write(
            all_mcs.get(), (jj + 1) * NUMBER_OF_P_BANDS,
            "TOTBIAS_MCS-" + std::to_string(nmc), jj + 1);
        monte_carlos_file.flush();

        converged = _syncMonteCarlo(
            nmc, raw_bias.get(), filt_bias.get(), NUMBER_OF_P_BANDS,
            "FTOTALBIAS");

        if (converged)
            break;
    }

    logTimings();
}


void Qu3DEstimator::testHSqrt() {
    constexpr int M_MCS = 5;
    double yTy = 0, xTHx = 0, xTx = 0;
    int status = 0;
    fitsfile *fits_file = nullptr;

    verbose = true;

    std::string out_fname = "!" + process::FNAME_BASE + "-testhqsrt-"
                            + std::to_string(mympi::this_pe) + ".fits";

    fits_create_file(&fits_file, out_fname.c_str(), &status);
    ioh::checkFitsStatus(status);

    /* define the name, datatype, and physical units for columns */
    int ncolumns = 2;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wwrite-strings"
    char *column_names[] = {"xTHx", "yTy"};
    char *column_types[] = {"1D", "1D"};
    #pragma GCC diagnostic pop

    fits_create_tbl(
        fits_file, BINARY_TBL, max_monte_carlos, ncolumns, column_names,
        column_types, nullptr, "HSQRT", &status);
    ioh::checkFitsStatus(status);

    auto xTx_arr = std::make_unique<double[]>(max_monte_carlos);
    auto yTy_arr = std::make_unique<double[]>(max_monte_carlos);
    auto xTHx_arr = std::make_unique<double[]>(max_monte_carlos);

    Progress prog_tracker(max_monte_carlos, 5);
    for (int i = 1; i <= max_monte_carlos; ++i) {
        yTy = 0;  xTHx = 0;  xTx = 0;

        #pragma omp parallel for schedule(dynamic, 4) reduction(+:xTx)
        for (auto &qso : quasars) {
            qso->blockRandom(rngs[myomp::getThreadNum()], p3d_model.get());
            std::copy_n(qso->truth, qso->N, qso->in);
            xTx += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
        }

        multiplyIpHVector(0.0);

        #pragma omp parallel for reduction(+:xTHx)
        for (auto &qso : quasars)
            xTHx += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);

        multiplyHsqrt();
        #pragma omp parallel for reduction(+:yTy)
        for (auto &qso : quasars)
            yTy += cblas_ddot(qso->N, qso->truth, 1, qso->truth, 1);

        xTx_arr[i - 1] = xTx;  xTHx_arr[i - 1] = xTHx;  yTy_arr[i - 1] = yTy;
        ++prog_tracker;

        verbose = false;
        if ((i % M_MCS != 0) && (i != max_monte_carlos))
            continue;

        int nelems = (i - 1) % M_MCS + 1, irow = i - nelems;
        fits_write_col(fits_file, TDOUBLE, 1, irow + 1, 1, nelems, xTHx_arr.get() + irow,
                       &status);
        fits_write_col(fits_file, TDOUBLE, 2, irow + 1, 1, nelems, yTy_arr.get() + irow,
                       &status);
        fits_write_col(fits_file, TDOUBLE, 3, irow + 1, 1, nelems, xTx_arr.get() + irow,
                       &status);
        fits_flush_buffer(fits_file, 0, &status);
        ioh::checkFitsStatus(status);
    }

    fits_close_file(fits_file, &status);
    ioh::checkFitsStatus(status);
}


void Qu3DEstimator::estimateNoiseBiasMc() {
    LOG::LOGGER.STD("Estimating noise bias.\n");
    verbose = false;
    mc1 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    mc2 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);

    Progress prog_tracker(max_monte_carlos, 10);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        /* generate random Gaussian vector into truth */
        #pragma omp parallel for
        for (auto &qso : quasars)
            rngs[myomp::getThreadNum()].fillVectorNormal(qso->truth, qso->N);
            // qso->fillRngNoise(rngs[myomp::getThreadNum()]);

        /* calculate Cinv . n into y */
        conjugateGradientDescent();
        /* Evolve with Z, save C^-1 . v (in) into mesh  & FFT */
        multiplyDerivVectors(mc1.get(), mc2.get());

        ++prog_tracker;

        if ((nmc % 5 != 0) && (nmc != max_monte_carlos))
            continue;

        converged = _syncMonteCarlo(
            nmc, raw_bias.get(), filt_bias.get(), NUMBER_OF_P_BANDS, "FBIAS");

        if (converged)
            break;
    }

    logTimings();
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
    LOG::LOGGER.STD("Estimating Fisher with random C,k.\n");
    verbose = false;
    mc1 = std::make_unique<double[]>(bins::FISHER_SIZE);
    mc2 = std::make_unique<double[]>(bins::FISHER_SIZE);
    timings["mDerivMatVec"] = std::make_pair(0, 0.0);

    /* Create another mesh to store random numbers. This way, randoms are
       generated once, and FFTd once. This grid can perform in-place FFTs,
       since it is only needed in Fourier space, which is equivalent between
       in-place and out-of-place transforms. */
    LOG::LOGGER.STD("  Constructing another mesh for randoms.\n");
    mesh_rnd.copy(mesh);
    mesh_rnd.initRngs(seed_generator.get());
    mesh_rnd.construct(INPLACE_FFT);

    // max_monte_carlos = 5;
    tolerance = 0.1;
    LOG::LOGGER.STD("  Using tolerance %lf.\n", tolerance);
    // max_conj_grad_steps = 1;
    // LOG::LOGGER.STD("  Using MaxConjGradSteps %d.\n", max_conj_grad_steps);
    Progress prog_tracker(max_monte_carlos, 5);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        mesh_rnd.fillRndOnes();
        mesh_rnd.fftX2K();
        LOG::LOGGER.STD("  Generated random numbers & FFT.\n");

        for (int i = 0; i < NUMBER_OF_P_BANDS; ++i) {
            multDerivMatrixVec(i);

            /* calculate C^-1 . qk into in */
            conjugateGradientDescent();
            /* Evolve with Z, save C^-1 . v (in) into mesh  & FFT */
            multiplyDerivVectors(mc1.get() + i * NUMBER_OF_P_BANDS,
                                 mc2.get() + i * NUMBER_OF_P_BANDS);
        }

        ++prog_tracker;

        // if ((nmc % 5 != 0) && (nmc != max_monte_carlos))
        //     continue;

        converged = _syncMonteCarlo(
            nmc, fisher.get(), covariance.get(), bins::FISHER_SIZE, "2FISHER");

        if (converged)
            break;
    }

    cblas_dscal(bins::FISHER_SIZE, 0.5, fisher.get(), 1);
    logTimings();
}


void Qu3DEstimator::estimateFisherDirect() {
    /* Tr[C^-1 . Qk . C^-1 . Qk'] = <z^T . C^-1 . Qk . C^-1 . Qk' . z>

    1. Generate z = +-1 per forest to *truth.
    2. Solve C^-1 . z to *in. Reverse interpolate to mesh_fh & FFT.
    3. Reverse interpolate init random (*truth) to mesh_rnd & FFT.
    4. Multiply init random (mesh_rnd) deriv mat. to *truth (through mesh).
    5. Solve C^-1 . Qk' . z.
    6. Multiply mesh_fh and mesh.
    */
    LOG::LOGGER.STD("Estimating Fisher directly.\n");
    verbose = false;
    mc1 = std::make_unique<double[]>(bins::FISHER_SIZE);
    mc2 = std::make_unique<double[]>(bins::FISHER_SIZE);
    timings["mDerivMatVec"] = std::make_pair(0, 0.0);

    LOG::LOGGER.STD("  Constructing two other meshes for randoms.\n");
    mesh_rnd.copy(mesh);  mesh_fh.copy(mesh);
    mesh_rnd.construct(INPLACE_FFT);  mesh_fh.construct(INPLACE_FFT);

    tolerance = 0.1;
    LOG::LOGGER.STD("  Using tolerance %lf.\n", tolerance);

    Progress prog_tracker(max_monte_carlos * NUMBER_OF_P_BANDS, 5);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        /* Generate z = +-1 per forest. */
        #pragma omp parallel for
        for (auto &qso : quasars)
            rngs[myomp::getThreadNum()].fillVectorOnes(qso->truth, qso->N);

        /* calculate C^-1 . z into in */
        conjugateGradientDescent();

        reverseInterpolateZ(mesh_fh);
        for (auto &qso : quasars)  std::swap(qso->in, qso->truth);
        reverseInterpolateZ(mesh_rnd);

        mesh_fh.rawFftX2K();  mesh_rnd.rawFftX2K();

        for (int i = 0; i < NUMBER_OF_P_BANDS; ++i) {
            multDerivMatrixVec(i);

            /* calculate C^-1 . qk into in */
            conjugateGradientDescent();
            multiplyDerivVectors(mc1.get() + i * NUMBER_OF_P_BANDS,
                                 mc2.get() + i * NUMBER_OF_P_BANDS,
                                 nullptr, mesh_fh);
            ++prog_tracker;
        }

        // if ((nmc % 5 != 0) && (nmc != max_monte_carlos))
        //     continue;

        converged = _syncMonteCarlo(
            nmc, fisher.get(), covariance.get(), bins::FISHER_SIZE, "2FISHER");

        if (converged)
            break;
    }

    cblas_dscal(bins::FISHER_SIZE, 0.5, fisher.get(), 1);
    logTimings();
}

#if 0
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
    mesh_rnd.convolveSqrtPk(p3d_model->interp2d_pL);

    if (pp_enabled) {
        /* Add small-scale clusting with conjugateGradientSampler
           This function fills truth with rng noise,
           Cy (out) with random vector with covariance S_s*/
        // conjugateGradientSampler();

        #pragma omp parallel for
        for (auto &qso : quasars) {
            qso->blockRandom(rngs[myomp::getThreadNum()], p3d_model.get());
            for (int i = 0; i < qso->N; ++i) {
                qso->truth[i] += mesh_rnd.interpolate(qso->r.get() + 3 * i);
                qso->truth[i] *= qso->isig[i] * qso->z1[i];
            }
            rngs[myomp::getThreadNum()].addVectorNormal(qso->truth, qso->N);
            std::copy_n(qso->truth, qso->N, qso->sc_eta);
            // std::swap(qso->truth, qso->sc_eta);
        }

        if (verbose)
            LOG::LOGGER.STD("Applying off-diagonal correlations.\n");

        for (int m = 1; m <= OFFDIAGONAL_ORDER; ++m) {
            double coeff = 1.0;
            for (int jj = 0; jj < m; ++jj)
                coeff *= (0.5 + jj) / (jj + 1);

            if (verbose)
                LOG::LOGGER.STD("  Order %d. Coefficient is %.4f.\n", m, coeff);
            // Solve C^-1 sc_eta into in
            conjugateGradientDescent();

            // multiply neighbors-only
            double t1_pp = mytime::timer.getTime(), dt_pp = 0;

            #pragma omp parallel for
            for (auto &qso : quasars) {
                if (qso->neighbors.empty())
                    continue;

                for (int i = 0; i < qso->N; ++i)
                    qso->in[i] *= qso->z1[i];
            }

            double nrm_truth_now = 0, nrm_sc_eta = 0;
            #pragma omp parallel for schedule(dynamic, 8) \
                reduction(+:nrm_truth_now, nrm_sc_eta)
            for (auto &qso : quasars) {
                if (qso->neighbors.empty())
                    std::fill_n(qso->out, qso->N, 0);
                else
                    qso->multCovNeighborsOnly(p3d_model.get(), effective_chi, qso->out);
                // *out is now N^-1/2 (S_OD C^-1)^m eta (BD random)

                // Truth is in *sc_eta
                nrm_truth_now += cblas_ddot(qso->N, qso->sc_eta, 1, qso->sc_eta, 1);

                qso->updateTruth(coeff);

                // *truth is now N^-1/2 (S_OD C^-1)^m eta (BD random)
                nrm_sc_eta += cblas_ddot(qso->N, qso->truth, 1, qso->truth, 1);
            }

            nrm_truth_now = sqrt(nrm_truth_now);
            nrm_sc_eta = coeff * sqrt(nrm_sc_eta);
            if (verbose)
                LOG::LOGGER.STD("Relative change: %.5e / %.5e = %.5e\n",
                    nrm_sc_eta, nrm_truth_now, nrm_sc_eta / nrm_truth_now);

            dt_pp = mytime::timer.getTime() - t1_pp;
            ++timings["PPcomp"].first;
            timings["PPcomp"].second += dt_pp;
        }

        for (auto &qso : quasars)
            std::swap(qso->truth, qso->sc_eta);
    }
    else {
        #pragma omp parallel for
        for (auto &qso : quasars) {
            qso->fillRngNoise(rngs[myomp::getThreadNum()]);
            for (int i = 0; i < qso->N; ++i)
                qso->truth[i] += qso->isig[i] * qso->z1[i]
                    * mesh_rnd.interpolate(qso->r.get() + 3 * i);
        }
    }

    t2 = mytime::timer.getTime() - t1;
    ++timings["GenGauss"].first;
    timings["GenGauss"].second += t2;

    if (verbose)
        LOG::LOGGER.STD("It took %.2f m.\n", t2);
}

void Qu3DEstimator::multiplyFisherDerivs(double *o1, double *o2) {
    static size_t mesh_kz_max = std::min(
        size_t(ceil((KMAX_EDGE - bins::KBAND_EDGES[0]) / mesh.k_fund[2])),
        mesh.ngrid_kz);
    static auto _lout = std::make_unique<double[]>(bins::FISHER_SIZE);
    static double *lout = _lout.get();

    double dt = mytime::timer.getTime();
    std::fill_n(lout, bins::FISHER_SIZE, 0);

    #pragma omp parallel for reduction(+:lout[0:bins::FISHER_SIZE])
    for (size_t jxy = 0; jxy < mesh.ngrid_xy; ++jxy) {
        double kperp = mesh.getKperpFromIperp(jxy),
               f_aa, f_ab, f_bb, temp;

        if (kperp >= KMAX_EDGE)
            continue;
        else if (kperp < specifics::MIN_KERP)
            continue;

        int ik = (kperp - bins::KBAND_EDGES[0]) / DK_BIN, ik2;

        size_t jj = mesh.ngrid_kz * jxy;
        kperp *= kperp;
        for (size_t k = 0; k < mesh_kz_max; ++k) {
            double kz = k * mesh.k_fund[2], kt = sqrt(kz * kz + kperp), mu;
            if (kt >= KMAX_EDGE || kt < bins::KBAND_EDGES[0])
                continue;

            ik = (kt - bins::KBAND_EDGES[0]) / DK_BIN;
            mu = kz / kt;

            f_aa = ((k != 0) + 1) * std::norm(mesh.field_k[k + jj])
                   * p3d_model->getSpectroWindow2(kz);
            temp = (1.0 - fabs(kt - bins::KBAND_CENTERS[ik]) / DK_BIN);
            f_bb = 1.0 - temp;
            f_ab = f_aa * temp * f_bb
            f_bb *= f_aa * f_bb;
            f_aa *= temp * temp;

            if (kt > bins::KBAND_CENTERS[ik])
                ik2 = std::min(bins::NUMBER_OF_K_BANDS - 1, ik + 1);
            else
                ik2 = std::max(0, ik - 1);

            for (int imu = 0; imu < NUMBER_OF_MULTIPOLES; ++imu) {
                int ii1 = ik + NUMBER_OF_K_BANDS * imu,
                    ii2 = ik2 + NUMBER_OF_K_BANDS * imu;

                double Lmu_i = legendre(2 * imu, mu);

                for (int jmu = imu; jmu < NUMBER_OF_MULTIPOLES; ++jmu) {
                    int jj1 = ik + NUMBER_OF_K_BANDS * jmu,
                        jj2 = ik2 + NUMBER_OF_K_BANDS * jmu;

                    double Lmu_j = Lmu_i * legendre(2 * jmu, mu);

                    lout[jj1 + ii1 * NUMBER_OF_P_BANDS] += Lmu_j * f_aa;
                    lout[jj2 + ii1 * NUMBER_OF_P_BANDS] += Lmu_j * f_ab;
                    lout[jj2 + jj2 * NUMBER_OF_P_BANDS] += Lmu_j * f_bb;
                }
            }
        }
    }

    cblas_dscal(bins::FISHER_SIZE, mesh.invtotalvol, lout, 1);

    if (o2 != nullptr) {
        for (int i = 0; i < bins::FISHER_SIZE; ++i) {
            o1[i] += lout[i];
            o2[i] += lout[i] * lout[i];
        }
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["mFisherDeriv"].first;
    timings["mFisherDeriv"].second += dt;
}
#endif