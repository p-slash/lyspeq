void Qu3DEstimator::multiplyAsVector(double s) {
    /* (I + N^-1/2 G^1/2 (S_S) G^1/2 N^-1/2) 
        input is const *in, output is *out
        uses: *in_isig, *sc_eta
    */
    double dt = mytime::timer.getTime();

    /* A_BD^-1. Might not be true if pp_enabled=false */
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        qso->setInIsigNoMarg();
        std::fill_n(qso->out, qso->N, 0);
    }

    // Add long wavelength mode to Cy
    // only in_isig is used until (B)
    // multMeshComp();

    if (pp_enabled)  multParticleComp();
    // (B)

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->out[i] *= qso->isig[i] * qso->z1[i];
            qso->out[i] += qso->in[i];
            qso->out[i] /= s;
        }
    }

    dt = mytime::timer.getTime() - dt;
    ++timings["mIpH"].first;
    timings["mIpH"].second += dt;
}


/* Multiply *in to *out */
void Qu3DEstimator::multiplyNewtonSchulzY(int n, double s) {
    if (n == 0) {
        multiplyAsVector(s);
        return;
    }

    // need temp storage
    std::vector<std::unique_ptr<double[]>> temp, cur_res;
    std::vector<double*> init_ins;
    temp.reserve(quasars.size());
    cur_res.reserve(quasars.size());
    init_ins.reserve(quasars.size());

    for (const auto &qso : quasars) {
        temp.push_back(std::make_unique<double[]>(qso->N));
        cur_res.push_back(std::make_unique<double[]>(qso->N));
        init_ins.push_back(qso->in);
    }

    // Get v1 to *out
    multiplyNewtonSchulzY(n - 1, s);

    // Save it to cur_res. Make it input of Z_n-1
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        std::copy_n(qso->out, qso->N, cur_res[i].get());
        qso->in = cur_res[i].get();
    }

    // Get Z . v1 to *out
    multiplyNewtonSchulzZ(n - 1, s);

    for (const auto &qso : quasars)
        temp.push_back(std::make_unique<double[]>(qso->N));

    // Make it input of Y_n-1
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        std::copy_n(qso->out, qso->N, temp[i].get());
        qso->in = temp[i].get();
    }

    // Get Y . Z . v1
    multiplyNewtonSchulzY(n - 1, s);
    temp.clear();

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        qso->in = init_ins[i];
        for (int j = 0; j < qso->N; ++j)
            qso->out[j] = 1.5 * cur_res[i][j] - 0.5 * qso->out[j];
    }
}


void Qu3DEstimator::multiplyNewtonSchulzZ(int n, double s) {
    if (n == 0) {
        for (auto &qso : quasars)
            std::copy_n(qso->in, qso->N, qso->out);
        return;
    }

    // need temp storage
    std::vector<std::unique_ptr<double[]>> temp, cur_res;
    std::vector<double*> init_ins;
    temp.reserve(quasars.size());
    cur_res.reserve(quasars.size());
    init_ins.reserve(quasars.size());

    for (const auto &qso : quasars) {
        temp.push_back(std::make_unique<double[]>(qso->N));
        cur_res.push_back(std::make_unique<double[]>(qso->N));
        init_ins.push_back(qso->in);
    }

    // Get z1 to *out
    multiplyNewtonSchulzZ(n - 1, s);

    // Save it to cur_res. Make it input of Z_n-1
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        std::copy_n(qso->out, qso->N, cur_res[i].get());
        qso->in = cur_res[i].get();
    }

    // Get Y . z1 to *out
    multiplyNewtonSchulzY(n - 1, s);

    for (const auto &qso : quasars)
        temp.push_back(std::make_unique<double[]>(qso->N));

    // Make it input of Y_n-1
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        std::copy_n(qso->out, qso->N, temp[i].get());
        qso->in = temp[i].get();
    }

    // Get Z . Y . z1
    multiplyNewtonSchulzZ(n - 1, s);
    temp.clear();

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        qso->in = init_ins[i];
        for (int j = 0; j < qso->N; ++j)
            qso->out[j] = 1.5 * cur_res[i][j] - 0.5 * qso->out[j];
    }
}


double Qu3DEstimator::estimateMaxEvalAs() {
    int niter = 1;
    double n_in, n_out, n_inout, new_eval_max, old_eval_max = 1e-12;
    bool is_converged = false, init_verbose = verbose;
    LOG::LOGGER.STD("Estimating maximum eigenvalue of As: ");

    std::vector<double*> init_ins;
    init_ins.reserve(quasars.size());
    for (const auto &qso : quasars)
        init_ins.push_back(qso->in);

    // find max_eval
    #pragma omp parallel for
    for (auto &qso : quasars) {
        rngs[myomp::getThreadNum()].fillVectorNormal(qso->sc_eta, qso->N);
        qso->in = qso->sc_eta;
    }

    verbose = false;
    for (; niter <= max_conj_grad_steps; ++niter) {
        multiplyAsVector();
        n_in = 0;  n_out = 0;  n_inout = 0;

        #pragma omp parallel for reduction(+:n_in, n_out, n_inout)
        for (const auto &qso : quasars) {
            n_in += cblas_ddot(qso->N, qso->in, 1, qso->in, 1);
            n_out += cblas_ddot(qso->N, qso->out, 1, qso->out, 1);
            n_inout += cblas_ddot(qso->N, qso->in, 1, qso->out, 1);
        }

        new_eval_max = n_inout / n_in;
        if (isClose(old_eval_max, new_eval_max, tolerance)) {
            is_converged = true;  break;
        }

        old_eval_max = new_eval_max;
        n_out = sqrt(n_out);
        for (auto &qso : quasars)
            for (int i = 0; i < qso->N; ++i)
                qso->in[i] = qso->out[i] / n_out;
    }

    if (is_converged)
        LOG::LOGGER.STD(" Converged: ");
    else
        LOG::LOGGER.STD(" NOT converged: ");

    LOG::LOGGER.STD(" %.5e (number of iterations: %d)\n", new_eval_max, niter);
    verbose = init_verbose;

    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        qso->in = init_ins[i];
    }
    return new_eval_max;
}


void Qu3DEstimator::multiplyCovSmallSqrtNewtonSchulz(int order) {
    static double max_eval = estimateMaxEvalAs();
    double s = (max_eval + 1.0) / 2.0;
    multiplyNewtonSchulzY(order, s);

    s = sqrt(s);
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] = qso->out[i] * s;
}
