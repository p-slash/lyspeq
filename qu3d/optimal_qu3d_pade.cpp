std::vector<double> _pade_xi(int pade_order) {
    std::vector<double> result;
    result.resize(pade_order);
    for (int i = 0; i < pade_order; ++i)
        result[i] = 0.5 * (1.0 + cos((2 * i + 1) * MY_PI / (2 * pade_order)));
    return result;
}


void Qu3DEstimator::multiplyCovSmallSqrtPade(int pade_order) {
    auto xi = _pade_xi(pade_order);
    std::vector<double> alphas;
    alphas.resize(xi.size());
    for (int i = 0; i < pade_order; ++i)
        alphas[i] = 1.0 / xi[i] - 1.0;

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        std::fill_n(qso->sc_eta, qso->N, 0);

    tolerance *= 10;
    if (verbose)
        LOG::LOGGER.STD("  Entered multiplyCovSmallSqrtPade with order %d. "
                        "New tolerance %.2e\n", pade_order, tolerance);
    for (int i = 0; i < pade_order; ++i) {
        conjugateGradientIpH(alphas[i]);
        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars)
            cblas_daxpy(qso->N, 1.0 / xi[i], qso->in, 1, qso->sc_eta, 1);
    }
    tolerance /= 10;

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        std::copy_n(qso->sc_eta, qso->N, qso->in);

    multiplyIpHVector(0);
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] = qso->out[i] / pade_order;
}
