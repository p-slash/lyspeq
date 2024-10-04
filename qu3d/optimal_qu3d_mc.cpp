void Qu3DEstimator::multiplyIpHVector(double m) {
    /* m I + I + N^-1/2 G^1/2 (S_L + S_OD) G^1/2 N^-1/2 A_BD^-1
        input is const *in, output is *out
        uses: *in_isig
    */
    double dt = mytime::timer.getTime();

    /* A_BD^-1. Might not be true if pp_enabled=false */
    #pragma omp parallel for schedule(dynamic, 8)
    for (auto &qso : quasars) {
        qso->multInvCov(p3d_model.get(), qso->in, qso->out, pp_enabled);
        std::swap(qso->in, qso->out);
        qso->setInIsigNoMarg();
        std::swap(qso->in, qso->out);
    }

    // Add long wavelength mode to Cy
    // only in_isig is used until (B)
    multMeshComp();

    if (pp_enabled) {
        #pragma omp parallel for schedule(dynamic, 8)
        for (auto &qso : quasars)
            qso->multCovNeighborsOnly(p3d_model.get(), effective_chi);
    }
    // (B)

    #pragma omp parallel for
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

    if (verbose)
        LOG::LOGGER.STD("  Entered conjugateGradientIpH.\n");

    /* Initial guess */
    #pragma omp parallel for schedule(dynamic, 8)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->in[i] = qso->truth[i] / 2.0;

    multiplyIpHVector(1.0);

    #pragma omp parallel for reduction(+:init_residual_norm, old_residual_prec)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->residual[i] = qso->truth[i] - qso->out[i];

        // Only search is multiplied from here until endconjugateGradientIpH
        qso->in = qso->search.get();

        // set search = InvCov . residual
        for (int i = 0; i < qso->N; ++i)
            qso->in[i] = qso->residual[i] / 2.0;

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
        // Calculate InvCov . residual into out
        #pragma omp parallel for reduction(+:new_residual_prec)
        for (auto &qso : quasars) {
            // set z (out) = InvCov . residual
            for (int i = 0; i < qso->N; ++i)
                qso->out[i] = qso->residual[i] / 2.0;

            new_residual_prec += cblas_ddot(qso->N, qso->residual.get(), 1, qso->out, 1);
        }

        double beta = new_residual_prec / old_residual_prec;
        old_residual_prec = new_residual_prec;

        // New direction using preconditioned z = InvCov . residual
        #pragma omp parallel for
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


void Qu3DEstimator::multiplyHsqrt() {
    /* multiply with SquareRootMatrix
        input is const *truth, output is *truth
        uses: *in, *in_isig, *sc_eta, *out
    */

    // (1) CG solve I + H from *truth to *in
    //     multiply *in with H, add to *sc_eta

    conjugateGradientIpH();

    multiplyIpHVector(0.0);

    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->sc_eta[i] += qso->out[i];
        std::swap(qso->truth, qso->in);
    }

    // (2) multiply *truth with 0.25 (I + H), add to sc_eta
    // for (auto &qso : quasars)
    //     std::swap(qso->truth, qso->in);

    multiplyIpHVector(1.0);

    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i)
            qso->sc_eta[i] += 0.25 * qso->out[i];
        std::swap(qso->truth, qso->sc_eta);
    }
}


void Qu3DEstimator::replaceDeltasWithGaussianField() {
    if (verbose)
        LOG::LOGGER.STD("Replacing deltas with Gaussian. ");

    double t1 = mytime::timer.getTime(), t2 = 0;
    #pragma omp parallel for
    for (auto &qso : quasars) {
        qso->blockRandom(rngs[myomp::getThreadNum()], p3d_model.get());
        std::copy_n(qso->truth, qso->N, qso->sc_eta);
    }

    multiplyHsqrt();

    t2 = mytime::timer.getTime() - t1;
    ++timings["GenGauss"].first;
    timings["GenGauss"].second += t2;

    if (verbose)
        LOG::LOGGER.STD("It took %.2f m.\n", t2);
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
#endif