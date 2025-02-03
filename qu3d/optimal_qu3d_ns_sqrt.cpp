/* Multiply *in to *out */
void Qu3DEstimator::multiplyNewtonSchulzY(int n, double s) {
    if (n == 0) {
        multiplyAsVector(0, s);
        return;
    }

    // need temp storage
    std::vector<std::unique_ptr<double[]>> temp, cur_res;
    std::vector<double*> init_ins;
    temp.reserve(quasars.size());
    cur_res.reserve(quasars.size());
    init_ins.reserve(quasars.size());

    for (const auto &qso : quasars) {
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


void Qu3DEstimator::multiplyCovSmallSqrtNewtonSchulz(int order) {
    static double max_eval = estimateMaxEvalAs();
    multiplyNewtonSchulzY(order, max_eval);

    double s = sqrt(max_eval);
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] = qso->out[i] * s;
}
