inline std::unique_ptr<double[]> _pade_xi(int pade_order) {
    auto result = std::make_unique<double[]>(pade_order);
    for (int i = 0; i < pade_order; ++i)
        result[i] = 0.5 * (1.0 + cos((2 * i + 1) * MY_PI / (2 * pade_order)));
    return result;
}


void Qu3DEstimator::multiplyCovSmallSqrtPade(int pade_order) {
    // static double max_eval = estimateMaxEvalAs();
    // static double min_eval = estimateMaxEvalAs(-max_eval);
    // if (min_eval < 0)
    //     throw std::runtime_error("Negative eigenvalue!");

    // double s = (min_eval + max_eval) / 2.0;
    // double s = 1.0;
    static double s = findMaxDiagonalAs();

    auto xi = std::make_unique<double[]>(pade_order),
         alphas = std::make_unique<double[]>(pade_order);

    for (int i = 0; i < pade_order; ++i) {
        xi[i] = 0.5 * (1.0 + cos((2 * i + 1) * MY_PI / (2 * pade_order)));
        xi[i] = 1.0 / xi[i];
        alphas[i] = xi[i] - 1.0;
        xi[i] *= sqrt(s) / pade_order;
    }

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        std::fill_n(qso->sc_eta, qso->N, 0);

    tolerance *= 10;
    if (verbose)
        LOG::LOGGER.STD("  Entered multiplyCovSmallSqrtPade with order %d. "
                        "New tolerance %.2e\n", pade_order, tolerance);

    for (int i = 0; i < pade_order; ++i) {
        conjugateGradientIpH(alphas[i], s);

        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars)
            cblas_daxpy(qso->N, xi[i], qso->in, 1, qso->sc_eta, 1);
    }
    tolerance /= 10;

    #pragma omp parallel for
    for (auto &qso : quasars)
        std::swap(qso->sc_eta, qso->in);

    multiplyAsVector(0, s);

    #pragma omp parallel for
    for (auto &qso : quasars) {
        std::swap(qso->sc_eta, qso->in);
        std::swap(qso->truth, qso->out);
    }
}
