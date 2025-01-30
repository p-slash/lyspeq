void Qu3DEstimator::multiplyIpHVector(double m) {
    /* m I + I + N^-1/2 G^1/2 (S_L + S_OD) G^1/2 N^-1/2 A_BD^-1
        input is const *in, output is *out
        uses: *in_isig, *sc_eta
    */
    double dt = mytime::timer.getTime();

    /* A_BD^-1. Might not be true if pp_enabled=false */
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        qso->multInvCov(p3d_model.get(), qso->in, qso->sc_eta, pp_enabled);
        double *tmp_in = qso->in;
        qso->in = qso->sc_eta;
        qso->setInIsigNoMarg();
        qso->in = tmp_in;
    }

    // Add long wavelength mode to Cy
    // only in_isig is used until (B)
    multMeshComp();

    if (pp_enabled)  multParticleComp();
    // (B)

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->out[i] *= qso->isig[i] * qso->z1[i];
            qso->out[i] += m * qso->in[i] + qso->sc_eta[i];
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

    constexpr double precon_diag = 2.0;
        // + (1.0 - p3d_model->getVar1dS() / p3d_model->getVar1dT());

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

        init_residual_norm += cblas_ddot(qso->N, qso->residual.get(), 1,
                                         qso->residual.get(), 1);
        old_residual_prec += cblas_ddot(qso->N, qso->residual.get(), 1,
                                        qso->in, 1);
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
        #pragma omp parallel for schedule(dynamic, 4) \
                                 reduction(+:new_residual_prec)
        for (auto &qso : quasars) {
            // set z (out) = PreCon . residual
            for (int i = 0; i < qso->N; ++i)
                qso->out[i] = qso->residual[i] / precon_diag;

            new_residual_prec += cblas_ddot(qso->N, qso->residual.get(), 1,
                                            qso->out, 1);
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
void Qu3DEstimator::multiplyCovSqrt() {
    /* multiply with SquareRootMatrix:
        0.25 A_BD^1/2 + H A_BD^1/2 (0.25 + (I+H)^-1)
        input is *truth, output is *truth
        uses: *in, *in_isig, *sc_eta, *out
    */

    // (1) CG solve I + H from *truth to *in
    //     add 0.25 *truth to *in
    // (2) multiply *in and *truth with A_BD^1/2
    conjugateGradientIpH();

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->in[i] += 0.25 * qso->truth[i];

        // (2) multiply *in and *truth with A_BD^1/2
        //     *truth <= 0.25 A_BD^1/2 *truth
        //     *in <= A_BD^1/2 *in
        qso->multSqrtCov(p3d_model.get());
    }

    // (3) multiply *in with H save to *out
    multiplyIpHVector(0.0);

    // (3) add *out to *truth (which is 0.25 A_BD^1/2 *truth(init))
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] += qso->out[i];
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
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->truth, qso->N);
        // qso->blockRandom(rngs[myomp::getThreadNum()], p3d_model.get());

    multiplyCovSqrt();

    t2 = mytime::timer.getTime() - t1;
    ++timings["GenGauss"].first;
    timings["GenGauss"].second += t2;

    if (verbose)
        LOG::LOGGER.STD("It took %.2f m.\n", t2);
}


void Qu3DEstimator::replaceDeltasWithHighResGaussianField() {
    LOG::LOGGER.STD("Replacing deltas with high-res. Gaussian. ");

    double t1 = mytime::timer.getTime();
    mesh_rnd.copy(mesh);
    for (int axis = 0; axis < 3; ++axis)
        mesh_rnd.ngrid[axis] *= mock_grid_res_factor;
    mesh_rnd.construct(INPLACE_FFT);
    mesh_rnd.fillRndNormal(rngs);
    mesh_rnd.convolveSqrtPk(p3d_model->interp2d_pT);
    double varlss = p3d_model->getVar1dT();

    std::vector<ioh::unique_fitsfile_ptr> file_writers;
    file_writers.reserve(myomp::getMaxNumThreads());
    for (int i = 0; i < myomp::getMaxNumThreads(); ++i) {
        std::string out_fname =
            "!" + process::FNAME_BASE + "-deltas-v"
            + std::to_string(mympi::this_pe) + "-" + std::to_string(i)
            + ".fits";
        file_writers.push_back(ioh::create_unique_fitsfile_ptr(out_fname));
    }

    #pragma omp parallel for schedule(static, 8)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            if (qso->isig[i] != 0) {
                qso->truth[i] =  qso->z1[i] * mesh_rnd.interpolate(
                    qso->r.get() + 3 * i
                    ) + rngs[myomp::getThreadNum()].normal() / qso->isig[i];
            }
            else
                qso->truth[i] = 0;

            // transform isig to ivar for project and write
            qso->isig[i] *= qso->isig[i];
        }

        // If project
        if (CONT_MARG_ENABLED)
            qso->project(varlss, specifics::CONT_LOGLAM_MARG_ORDER);

        // Save to deltas
        qso->write(file_writers[myomp::getThreadNum()].get());
    }

    LOG::LOGGER.STD("It took %.2f m.\n", mytime::timer.getTime() - t1);
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
            "MC converges when < %.2e\n", nmc, mean_std, max_std, mc_tol);

        converged = max_std < mc_tol;
    }

    mympi::bcast(&converged);
    return converged;
}


void Qu3DEstimator::estimateTotalBiasDirect() {
    constexpr int M_MCS = 5;
    verbose = false;
    mc1 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    mc2 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    int nmc = 1;
    bool converged = false;

    if (!mesh_rnd) {mesh_rnd.copy(mesh); mesh_rnd.construct(INPLACE_FFT);}

    // Direct estimation first.
    LOG::LOGGER.STD("Estimating total bias directly.\n");
    Progress prog_tracker(max_monte_carlos, 10);
    for (; nmc <= max_monte_carlos; ++nmc) {
        /* Generate z = +-1 per forest. */
        #pragma omp parallel for
        for (auto &qso : quasars)
            rngs[myomp::getThreadNum()].fillVectorOnes(qso->in, qso->N);

        /* (Right hand side) Save this on mesh_rnd */
        reverseInterpolateZ(mesh_rnd);
        mesh_rnd.rawFftX2K();

        /* (Left hand side) CGD requires *truth to be mult'd by N^-1/2
           Swapping ptrs could be buggy:
                qso->multIsigInVector();
                std::swap(qso->in, qso->truth);
        */
        #pragma omp parallel for
        for (auto &qso : quasars)
            mxhelp::vector_multiply(qso->N, qso->in, qso->isig, qso->truth);

        /* calculate C^-1 . z into *in */
        conjugateGradientDescent();

        multiplyDerivVectors(mc1.get(), mc2.get(), nullptr, mesh_rnd);
        ++prog_tracker;

        if ((nmc % M_MCS != 0) && (nmc != max_monte_carlos))
            continue;

        converged = _syncMonteCarlo(
            nmc, raw_bias.get(), filt_bias.get(), NUMBER_OF_P_BANDS,
            "FTOTALBIAS-D");

        if (converged)
            break;
    }

    logTimings();
}

void Qu3DEstimator::estimateTotalBiasMc() {
    /* Saves every Monte Carlo simulation. The results need to be
       post-processed to get the Fisher matrix. */
    constexpr int M_MCS = 5;
    verbose = false;
    mc1 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    mc2 = std::make_unique<double[]>(NUMBER_OF_P_BANDS);
    int nmc = 1;
    bool converged = false;

    // Monte Carlo estimation.
    LOG::LOGGER.STD("Estimating total bias with Monte Carlos.\n");
    // Every task saves their own Monte Carlos
    ioh::Qu3dFile monte_carlos_file(
        process::FNAME_BASE + "-montecarlos-" + std::to_string(mympi::this_pe),
        0);
    auto all_mcs = std::make_unique<double[]>(M_MCS * NUMBER_OF_P_BANDS);

    Progress prog_tracker(max_monte_carlos, 10);
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
            "FTOTALBIAS-MC");

        if (converged)
            break;
    }

    logTimings();
}


void Qu3DEstimator::testCovSqrt() {
    constexpr int M_MCS = 5;
    double yTy = 0, xTHx = 0, xTx = 0;
    int status = 0;

    verbose = true;

    std::string out_fname = "!" + process::FNAME_BASE + "-testhqsrt-"
                            + std::to_string(mympi::this_pe) + ".fits";
    auto fitsfile_ptr = ioh::create_unique_fitsfile_ptr(out_fname);
    fitsfile *fits_file = fitsfile_ptr.get();

    /* define the name, datatype, and physical units for columns */
    int ncolumns = 3;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wwrite-strings"
    char *column_names[] = {"xTHx", "yTy", "xTx"};
    char *column_types[] = {"1D", "1D", "1D"};
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
            rngs[myomp::getThreadNum()].fillVectorNormal(qso->truth, qso->N);
            // qso->blockRandom(rngs[myomp::getThreadNum()], p3d_model.get());
            std::copy_n(qso->truth, qso->N, qso->in);
            xTx += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
        }

        multiplyCovVector();

        #pragma omp parallel for reduction(+:xTHx)
        for (auto &qso : quasars)
            xTHx += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);

        multiplyCovSqrt();
        #pragma omp parallel for reduction(+:yTy)
        for (auto &qso : quasars)
            yTy += cblas_ddot(qso->N, qso->truth, 1, qso->truth, 1);

        xTx_arr[i - 1] = xTx;  xTHx_arr[i - 1] = xTHx;  yTy_arr[i - 1] = yTy;
        ++prog_tracker;

        verbose = false;
        if ((i % M_MCS != 0) && (i != max_monte_carlos))
            continue;

        int nelems = (i - 1) % M_MCS + 1, irow = i - nelems;
        fits_write_col(fits_file, TDOUBLE, 1, irow + 1, 1, nelems,
                       xTHx_arr.get() + irow, &status);
        fits_write_col(fits_file, TDOUBLE, 2, irow + 1, 1, nelems,
                       yTy_arr.get() + irow, &status);
        fits_write_col(fits_file, TDOUBLE, 3, irow + 1, 1, nelems,
                       xTx_arr.get() + irow, &status);
        fits_flush_buffer(fits_file, 0, &status);
        ioh::checkFitsStatus(status);
    }

    logTimings();
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
    mesh_rnd.construct(INPLACE_FFT);

    // max_monte_carlos = 5;
    double _old_tolerace = tolerance;
    tolerance = 0.1;
    LOG::LOGGER.STD("  Using tolerance %lf.\n", tolerance);
    // max_conj_grad_steps = 1;
    // LOG::LOGGER.STD("  Using MaxConjGradSteps %d.\n", max_conj_grad_steps);
    Progress prog_tracker(max_monte_carlos, 5);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        mesh_rnd.fillRndOnes(rngs);
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
    tolerance = _old_tolerace;
}


void Qu3DEstimator::estimateFisherDirect() {
    /* Tr[C^-1 . Qk . C^-1 . Qk'] = <z^T . C^-1 . Qk . C^-1 . Qk' . z>

    1. Generate z = +-1 per forest to *truth.
    2. Reverse interpolate init random (*truth) to mesh_rnd & FFT.
    3. Solve C^-1 . z to *in. Reverse interpolate to mesh_fh & FFT.
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
    if (!mesh_rnd) { mesh_rnd.copy(mesh); mesh_rnd.construct(INPLACE_FFT); }
    if (!mesh_fh) { mesh_fh.copy(mesh); mesh_fh.construct(INPLACE_FFT); }

    LOG::LOGGER.STD("  Using preconditioner as solution.\n");

    Progress prog_tracker(max_monte_carlos * NUMBER_OF_P_BANDS, 5);
    int nmc = 1;
    bool converged = false;
    for (; nmc <= max_monte_carlos; ++nmc) {
        /* Generate z = +-1 per forest. */
        #pragma omp parallel for
        for (auto &qso : quasars)
            rngs[myomp::getThreadNum()].fillVectorOnes(qso->in, qso->N);

        /* (Right hand side) Save this on mesh_rnd */
        reverseInterpolateZ(mesh_rnd);
        mesh_rnd.rawFftX2K();

        /* (Left hand side ) CGD requires *truth to be multiplied by N^-1/2 */
        #pragma omp parallel for
        for (auto &qso : quasars)
            mxhelp::vector_multiply(qso->N, qso->in, qso->isig, qso->truth);

        /* calculate C^-1 . z into *in */
        preconditionerSolution();

        reverseInterpolateZ(mesh_fh);
        mesh_fh.rawFftX2K();

        for (int i = 0; i < NUMBER_OF_P_BANDS; ++i) {
            multDerivMatrixVec(i);

            /* calculate C^-1 . qk into in */
            preconditionerSolution();
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
