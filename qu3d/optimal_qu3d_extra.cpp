double Qu3DEstimator::estimateFrobeniusNormAs() {
    // Trace (C C)
    constexpr int M_MCS = 5;
    verbose = false;
    int nmc = 1;
    bool is_converged = false;
    double norm_F = -1, cur_total = 0;

    // Direct estimation first.
    LOG::LOGGER.STD("Estimating the Frobenius norm...");
    for (; nmc <= max_conj_grad_steps; ++nmc) {
        /* Generate z = +-1 per forest. */
        #pragma omp parallel for
        for (auto &qso : quasars)
            rngs[myomp::getThreadNum()].fillVectorOnes(qso->in, qso->N);

        multiplyAsVector();

        #pragma omp parallel for reduction(+:cur_total)
        for (auto &qso : quasars)
            cur_total += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);

        if ((nmc % M_MCS != 0) && (nmc != max_monte_carlos))
            continue;

        double cur_norm = sqrt(cur_total / nmc);
        if (isClose(cur_norm, norm_F, tolerance)) {
            is_converged = true;
            norm_F = cur_norm;
            break;
        }

        norm_F = cur_norm;
    }

    if (is_converged)
        LOG::LOGGER.STD(" Converged (%d iterations): %.2e\n", nmc, norm_F);
    else
        LOG::LOGGER.STD(" NOT converged: %.2e\n", norm_F);

    return norm_F;
}

void Qu3DEstimator::calculateNewDirection(double beta)  {
    #pragma omp parallel for
    for (auto &qso : quasars) {
        #pragma omp simd
        for (int i = 0; i < qso->N; ++i)
            qso->search[i] = beta * qso->search[i] + qso->residual[i];
    }
}


double Qu3DEstimator::updateRng(double residual_norm2) {
    /* This is called only for small-scale direct multiplication
       in conjugateGradientSampler. */
    double t1 = mytime::timer.getTime(), t2 = 0;

    double pTCp = 0, alpha = 0, new_residual_norm2 = 0;

    /* Multiply C x search into Cy => C. p(in) = out*/
    #pragma omp parallel for
    for (auto &qso : quasars)
        std::fill_n(qso->out, qso->N, 0);

    multParticleComp();

    // Get pT . C . p
    #pragma omp parallel for reduction(+:pTCp)
    for (const auto &qso : quasars)
        pTCp += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);

    if (pTCp <= 0)  return 0;

    alpha = residual_norm2 / pTCp;

    double zrnd = rngs[0].normal() / sqrt(pTCp);

    #pragma omp parallel for reduction(+:new_residual_norm2)
    for (auto &qso : quasars) {
        /* in is search.get() */
        cblas_daxpy(qso->N, zrnd, qso->in, 1, qso->y.get(), 1);
        cblas_daxpy(qso->N, -alpha, qso->out, 1, qso->residual.get(), 1);

        new_residual_norm2 += cblas_ddot(
            qso->N, qso->residual.get(), 1, qso->residual.get(), 1);
    }

    t2 = mytime::timer.getTime() - t1;
    if (verbose)
        LOG::LOGGER.STD("    updateRng took %.2f s.\n", 60.0 * t2);

    return new_residual_norm2;
}


void Qu3DEstimator::conjugateGradientSampler() {
    double dt = mytime::timer.getTime();
    int niter = 1;

    if (verbose)
        LOG::LOGGER.STD("  Entered conjugateGradientSampler.\n");

    double old_residual_norm2 = 0, init_residual_norm = 0;

    /* Truth is always random. Initial guess is always zero */
    #pragma omp parallel for reduction(+:old_residual_norm2)
    for (auto &qso : quasars) {
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->truth, qso->N);
        // qso->fillRngNoise(rngs[myomp::getThreadNum()]);
        std::copy_n(qso->truth, qso->N, qso->residual.get());
        std::copy_n(qso->truth, qso->N, qso->search.get());
        std::fill_n(qso->in, qso->N, 0);
        // Only seached multiplied from here until endconjugateGradientSampler
        qso->in = qso->search.get();
        qso->in_isig = qso->in;

        old_residual_norm2 += cblas_ddot(
            qso->N, qso->residual.get(), 1, qso->residual.get(), 1);
    }

    init_residual_norm = sqrt(old_residual_norm2);

    if (hasConverged(init_residual_norm, tolerance))
        goto endconjugateGradientSampler;

    if (absolute_tolerance) init_residual_norm = 1;

    for (; niter <= max_conj_grad_steps; ++niter) {
        double new_residual_norm2 = updateRng(old_residual_norm2);

        bool end_iter = hasConverged(
            sqrt(new_residual_norm2) / init_residual_norm, tolerance);

        if (end_iter)
            goto endconjugateGradientSampler;

        double beta = new_residual_norm2 / old_residual_norm2;
        old_residual_norm2 = new_residual_norm2;
        calculateNewDirection(beta);
    }

endconjugateGradientSampler:
    if (verbose)
        LOG::LOGGER.STD(
            "  conjugateGradientSampler finished in %d iterations.\n", niter);

    /* Multiply y with S_s, save to out, restore in_isig to y_isig.get() */
    for (auto &qso : quasars) {
        qso->in = qso->y.get();
        qso->in_isig = qso->in;
        std::fill_n(qso->out, qso->N, 0);
    }

    multParticleComp();

    for (auto &qso : quasars)
        qso->in_isig = qso->y_isig.get();

    dt = mytime::timer.getTime() - dt;
    ++timings["CGS"].first;
    timings["CGS"].second += dt;
}


/* void Qu3DEstimator::cgsGetY() {
    if (CONT_MARG_ENABLED) {
        #pragma omp parallel for schedule(dynamic, 8)
        for (auto &qso : quasars) {
            qso->in = qso->y.get();
            std::swap(qso->truth, qso->in);
            qso->multTruthWithMarg();
            std::swap(qso->truth, qso->in);
            qso->multIsigInVector();
        }
    }
    else {
        #pragma omp parallel for
        for (auto &qso : quasars) {
            qso->in = qso->y.get();
            qso->multIsigInVector();
        }
    }
} */


void Qu3DEstimator::dumpSearchDirection() {
    if (mympi::this_pe != 0)
        return;

    int status = 0;
    std::string out_fname = "!" + process::FNAME_BASE + "-dump-search.fits";

    auto fitsfile_ptr = ioh::create_unique_fitsfile_ptr(out_fname);
    fitsfile *fits_file = fitsfile_ptr.get();

    /* define the name, datatype, and physical units for columns */
    int ncolumns = 4;
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wwrite-strings"
    char *column_names[] = {"Gz", "Chi", "Search", "AxSearch"};
    char *column_types[] = {"1D", "1E", "1D", "1D"};
    char *column_units[] = { "A", "Mpc", "\0", "\0"};
    #pragma GCC diagnostic pop

    for (const auto &qso : quasars) {
        int nrows = qso->N;
        auto chi = std::make_unique<float[]>(nrows);
        for (int i = 0; i < nrows; ++i)
            chi[i] = qso->r[3 * i + 2];

        fits_create_tbl(
            fits_file, BINARY_TBL, nrows, ncolumns, column_names, column_types,
            column_units, std::to_string(qso->qFile->id).c_str(), &status);
        ioh::checkFitsStatus(status);
        fits_write_key(
            fits_file, TDOUBLE, "RA", &qso->angles[0], nullptr, &status);
        fits_write_key(
            fits_file, TDOUBLE, "DEC", &qso->angles[1], nullptr, &status);
        fits_write_key(
            fits_file, TDOUBLE, "MEAN_SNR", &qso->qFile->snr, nullptr, &status);
        int nmbrs = qso->neighbors.size();
        fits_write_key(fits_file, TINT, "NUM_NEIG", &nmbrs, nullptr, &status);
        fits_write_col(fits_file, TDOUBLE, 1, 1, 1, nrows, qso->z1, &status);
        fits_write_col(fits_file, TFLOAT, 2, 1, 1, nrows, chi.get(), &status);
        fits_write_col(
            fits_file, TDOUBLE, 3, 1, 1, nrows, qso->search.get(), &status);
        fits_write_col(fits_file, TDOUBLE, 4, 1, 1, nrows, qso->out, &status);
        ioh::checkFitsStatus(status);
    }
}


void Qu3DEstimator::testSymmetry() {
    double uTAv = 0, vTAu = 0;
    verbose = false;

    #pragma omp parallel for
    for (auto &qso : quasars) {
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->in, qso->N);
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->truth, qso->N);
        qso->setInIsigNoMarg();
    }

    multMeshComp();
    #pragma omp parallel for reduction(+:uTAv)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->out[i] *= qso->isig[i] * qso->z1[i];
        uTAv += cblas_ddot(qso->N, qso->truth, 1, qso->out, 1);
    }
    LOG::LOGGER.STD("uTS_Lv = %.9e, ", uTAv);

    #pragma omp parallel for
    for (auto &qso : quasars) {
        std::swap(qso->in, qso->truth);
        qso->setInIsigNoMarg();
    }

    multMeshComp();
    #pragma omp parallel for reduction(+:vTAu)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->out[i] *= qso->isig[i] * qso->z1[i];
        vTAu += cblas_ddot(qso->N, qso->truth, 1, qso->out, 1);
    }
    LOG::LOGGER.STD("vTS_Lu = %.9e. Diff: %.9e\n", vTAu, vTAu - uTAv);

    // Small scale
    uTAv = 0;  vTAu = 0;
    #pragma omp parallel for schedule(dynamic, 4) reduction(+:uTAv)
    for (auto &qso : quasars) {
        std::fill_n(qso->out, qso->N, 0);
        qso->multCovNeighbors(p3d_model.get(), effective_chi);
        for (int i = 0; i < qso->N; ++i)
            qso->out[i] *= qso->isig[i] * qso->z1[i];
        uTAv += cblas_ddot(qso->N, qso->truth, 1, qso->out, 1);
    }
    LOG::LOGGER.STD("uTS_Sv = %.9e, ", uTAv);

    #pragma omp parallel for
    for (auto &qso : quasars) {
        std::swap(qso->in, qso->truth);
        qso->setInIsigNoMarg();
    }

    #pragma omp parallel for schedule(dynamic, 4) reduction(+:vTAu)
    for (auto &qso : quasars) {
        std::fill_n(qso->out, qso->N, 0);
        qso->multCovNeighbors(p3d_model.get(), effective_chi);
        for (int i = 0; i < qso->N; ++i)
            qso->out[i] *= qso->isig[i] * qso->z1[i];
        vTAu += cblas_ddot(qso->N, qso->truth, 1, qso->out, 1);
    }
    LOG::LOGGER.STD("vTS_Su = %.9e. Diff: %.9e\n", vTAu, vTAu - uTAv);
}



void Qu3DEstimator::estimateMaxEvals() {
    int niter = 1;
    double n_in, n_out, n_inout, new_eval_max, old_eval_max = 1e-12;
    verbose = false;

    LOG::LOGGER.STD("Estimating maximum eigenvalue of C.\n");

/*
    #pragma omp parallel for
    for (auto &qso : quasars)
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->in, qso->N);

    for (; niter <= max_conj_grad_steps; ++niter) {
        multiplyCovVector();
        n_in = 0;  n_out = 0;  n_inout = 0;

        #pragma omp parallel for reduction(+:n_in, n_out, n_inout)
        for (const auto &qso : quasars) {
            n_in += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
            n_out += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);
            n_inout += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);
        }

        new_eval_max = n_inout / n_in;
        LOG::LOGGER.STD("  New eval: %.5e\n", new_eval_max);
        if (isClose(old_eval_max, new_eval_max, tolerance)) {
            LOG::LOGGER.STD("Converged.\n");  break;
        }

        old_eval_max = new_eval_max;
        n_out = sqrt(n_out);
        for (auto &qso : quasars)
            for (int i = 0; i < qso->N; ++i)
                qso->in[i] = qso->out[i] / n_out;
    }

    LOG::LOGGER.STD("Estimating maximum eigenvalue of S_OD.\n");

    niter = 1;
    old_eval_max = 1e-12;
    #pragma omp parallel for
    for (auto &qso : quasars)
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->in, qso->N);

    for (; niter <= max_conj_grad_steps; ++niter) {
        #pragma omp parallel for
        for (auto &qso : quasars) {
            if (qso->neighbors.empty())
                continue;

            for (int i = 0; i < qso->N; ++i)
                qso->in[i] *= qso->z1[i];
        }

        n_in = 0;  n_out = 0;  n_inout = 0;
        #pragma omp parallel for schedule(dynamic, 8) \
                                 reduction(+:n_in, n_out, n_inout)
        for (auto &qso : quasars) {
            std::fill_n(qso->out, qso->N, 0);
            if (!qso->neighbors.empty())
                qso->multCovNeighborsOnly(p3d_model.get(), effective_chi);

            n_in += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
            n_out += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);
            n_inout += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);
        }

        new_eval_max = n_inout / n_in;
        LOG::LOGGER.STD("  New eval: %.5e\n", new_eval_max);
        if (isClose(old_eval_max, new_eval_max, tolerance)) {
            LOG::LOGGER.STD("Converged.\n");  break;
        }

        old_eval_max = new_eval_max;
        n_out = sqrt(n_out);

        #pragma omp parallel for
        for (auto &qso : quasars)
            for (int i = 0; i < qso->N; ++i)
                qso->in[i] = qso->out[i] / n_out;
    }

    LOG::LOGGER.STD("Estimating maximum eigenvalue of C inverse.\n");

    niter = 1;
    old_eval_max = 1e-12;
    #pragma omp parallel for
    for (auto &qso : quasars)
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->sc_eta, qso->N);

    for (; niter <= max_conj_grad_steps; ++niter) {
        #pragma omp parallel for
        for (auto &qso : quasars)
            for (int i = 0; i < qso->N; ++i)
                qso->truth[i] = qso->sc_eta[i] * qso->isig[i];

        conjugateGradientDescent();

        n_in = 0;  n_out = 0;  n_inout = 0;
        #pragma omp parallel for reduction(+:n_in, n_out, n_inout)
        for (auto &qso : quasars) {
            n_in += cblas_ddot(qso->N, qso->sc_eta, 1, qso->sc_eta, 1);
            n_out += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
            n_inout += cblas_ddot(qso->N, qso->sc_eta, 1, qso->in, 1);
        }

        new_eval_max = n_inout / n_in;
        LOG::LOGGER.STD("  New eval: %.5e\n", new_eval_max);
        if (isClose(old_eval_max, new_eval_max, tolerance, 1e-15)) {
            LOG::LOGGER.STD("Converged.\n");  break;
        }

        old_eval_max = new_eval_max;
        n_out = sqrt(n_out);

        #pragma omp parallel for
        for (auto &qso : quasars)
            for (int i = 0; i < qso->N; ++i)
                qso->sc_eta[i] = qso->in[i] / n_out;
    }

    LOG::LOGGER.STD("Estimating maximum eigenvalue of S_OD C^-1.\n");

    niter = 1;
    old_eval_max = 1e-12;
    #pragma omp parallel for
    for (auto &qso : quasars)
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->truth, qso->N);

    for (; niter <= max_conj_grad_steps; ++niter) {
        conjugateGradientDescent();

        #pragma omp parallel for
        for (auto &qso : quasars) {
            if (qso->neighbors.empty())
                continue;

            for (int i = 0; i < qso->N; ++i)
                qso->in[i] *= qso->z1[i];
        }

        n_in = 0;  n_out = 0;  n_inout = 0;
        #pragma omp parallel for schedule(dynamic, 8) \
                                 reduction(+:n_in, n_out, n_inout)
        for (auto &qso : quasars) {
            std::fill_n(qso->out, qso->N, 0);
            if (!qso->neighbors.empty())
                qso->multCovNeighborsOnly(p3d_model.get(), effective_chi);

            n_in += cblas_ddot(qso->N, qso->truth, 1, qso->truth, 1);
            n_out += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);
            n_inout += cblas_ddot(qso->N, qso->truth, 1, qso->out, 1);
        }

        new_eval_max = n_inout / n_in;
        LOG::LOGGER.STD("  New eval: %.5e\n", new_eval_max);
        if (isClose(old_eval_max, new_eval_max, tolerance)) {
            LOG::LOGGER.STD("Converged.\n");  break;
        }

        old_eval_max = new_eval_max;
        n_out = sqrt(n_out);

        #pragma omp parallel for
        for (auto &qso : quasars)
            for (int i = 0; i < qso->N; ++i)
                qso->truth[i] = qso->out[i] / n_out;
    }
*/
    LOG::LOGGER.STD("Estimating maximum eigenvalue of V A_BD^-1.\n");

    niter = 1;
    old_eval_max = 1e-12;
    #pragma omp parallel for
    for (auto &qso : quasars)
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->in, qso->N);

    for (; niter <= max_conj_grad_steps; ++niter) {
        multiplyAsVector();

        n_in = 0;  n_out = 0;  n_inout = 0;
        #pragma omp parallel for schedule(dynamic, 8) \
                                 reduction(+:n_in, n_out, n_inout)
        for (auto &qso : quasars) {
            n_in += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
            n_out += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);
            n_inout += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);
        }

        new_eval_max = n_inout / n_in;
        LOG::LOGGER.STD("  New eval: %.5e\n", new_eval_max);
        if (isClose(old_eval_max, new_eval_max, tolerance)) {
            LOG::LOGGER.STD("Converged.\n");  break;
        }

        old_eval_max = new_eval_max;
        n_out = sqrt(n_out);

        #pragma omp parallel for
        for (auto &qso : quasars)
            for (int i = 0; i < qso->N; ++i)
                qso->in[i] = qso->out[i] / n_out;
    }
}
