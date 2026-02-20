inline std::unique_ptr<double[]> _compute_pade_alphas(int order) {
    auto alphas = std::make_unique<double[]>(order);
    for (int i = 0; i < order; ++i) {
        alphas[i] = 0.5 * (1.0 + cos((2 * i + 1) * MY_PI / (2 * order)));
        alphas[i] = 1.0 / alphas[i] - 1.0;
    }
    return alphas;
}

inline std::unique_ptr<double[]> _compute_pade_xi(
        const std::unique_ptr<double[]> &alphas, double shrink, int order
) {
    auto xi = std::make_unique<double[]>(order);
    for (size_t i = 0; i < order; ++i) {
        xi[i] = (1.0 + alphas[i]) * sqrt(shrink) / order;
    }
    return xi;
}

inline void print_alphas_xi(
        const double* alphas, const double* xi, int order
) {
    std::string result = "  alphas: ";
    char buf[32];
    for (int i = 0; i < order; ++i) {
        std::snprintf(buf, sizeof(buf), "%.2e ", alphas[i]);
        result += buf;
    }
    result += "\n  xi    : ";
    for (int i = 0; i < order; ++i) {
        std::snprintf(buf, sizeof(buf), "%.2e ", xi[i]);
        result += buf;
    }
    LOG::LOGGER.STD("%s\n", result.c_str());
}

void Qu3DEstimator::multiplyCovSmallSqrtPade() {
    // static double max_eval = estimateMaxEvalAs();
    // static double min_eval = estimateMaxEvalAs(-max_eval);
    // if (min_eval < 0)
    //     throw std::runtime_error("Negative eigenvalue!");

    // double s = (min_eval + max_eval) / 2.0;
    // double s = 1.0;
    tolerance *= 10;
    double mp = 1.0 - mixture_factor_for_as;

    if (shrink_factor_for_sqrt == 0.0)
        shrink_factor_for_sqrt = findMaxDiagonalAs();
    else if (shrink_factor_for_sqrt == -1.0)
        shrink_factor_for_sqrt = estimateMaxEvalAs(-mp);
    else if (shrink_factor_for_sqrt == -2.0)
        shrink_factor_for_sqrt = estimateMaxEvalAs(-mp, true);
    else if (shrink_factor_for_sqrt == -3.0)
        shrink_factor_for_sqrt = estimateFrobeniusNormAs(-mp);
    
    static auto alphas = _compute_pade_alphas(pade_order);
    static auto xi = [this]() {
        auto ptr = std::make_unique<double[]>(pade_order);
        for (size_t i = 0; i < pade_order; ++i) {
            ptr[i] = (1.0 + alphas[i]) * sqrt(shrink_factor_for_sqrt) / pade_order;
        }
        return ptr;
    }();

    if (verbose) {
        LOG::LOGGER.STD(
            "  Entered multiplyCovSmallSqrtPade with order %d. "
            "Shriking factor %.5f. New tolerance %.2e.\n",
            pade_order, shrink_factor_for_sqrt, tolerance);
        print_alphas_xi(alphas.get(), xi.get(), pade_order);
    }

    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        std::fill_n(qso->sc_eta, qso->N, 0);

    for (int i = 0; i < pade_order; ++i) {
        conjugateGradientIpH(alphas[i], shrink_factor_for_sqrt);

        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars)
            cblas_daxpy(qso->N, xi[i], qso->in, 1, qso->sc_eta, 1);
    }
    tolerance /= 10;

    #pragma omp parallel for
    for (auto &qso : quasars)
        std::swap(qso->sc_eta, qso->in);

    multiplyAsVector(0, shrink_factor_for_sqrt);

    #pragma omp parallel for
    for (auto &qso : quasars) {
        std::swap(qso->sc_eta, qso->in);
        std::swap(qso->truth, qso->out);
    }
}
