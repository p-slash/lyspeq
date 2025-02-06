int MAX_TEMP_ARRAYS_SCHULZ = 0, CURR_NUM_TEMP_ARRAYS_SCHULZ = 0;
int NUMBER_SCHULZ_MULTIS = 0;

/* Multiply *in to *out */
void Qu3DEstimator::multiplyNewtonSchulzY(int n, double s) {
    if (n == 0) {
        multiplyAsVector(0, s);
        ++NUMBER_SCHULZ_MULTIS;
        return;
    }
    else if (n == 1) {
        multiplyAsVector(0, s);
        ++NUMBER_SCHULZ_MULTIS;

        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars) {
            for (int j = 0; j < qso->N; ++j)
                qso->sc_eta[j] = 1.5 * qso->in[j] - 0.5 * qso->out[j];
            std::swap(qso->sc_eta, qso->in);
        }

        multiplyAsVector(0, s);
        ++NUMBER_SCHULZ_MULTIS;

        for (auto &qso : quasars)
            std::swap(qso->sc_eta, qso->in);
        return;
    }

    // need temp storage
    std::vector<std::unique_ptr<double[]>> temp, cur_res;
    std::vector<double*> init_ins;
    cur_res.reserve(quasars.size());
    init_ins.reserve(quasars.size());

    for (const auto &qso : quasars) {
        cur_res.push_back(std::make_unique<double[]>(qso->N));
        init_ins.push_back(qso->in);
    }
    ++CURR_NUM_TEMP_ARRAYS_SCHULZ;
    MAX_TEMP_ARRAYS_SCHULZ = std::max(
        MAX_TEMP_ARRAYS_SCHULZ, CURR_NUM_TEMP_ARRAYS_SCHULZ);

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

    temp.reserve(quasars.size());
    for (const auto &qso : quasars)
        temp.push_back(std::make_unique<double[]>(qso->N));
    ++CURR_NUM_TEMP_ARRAYS_SCHULZ;
    MAX_TEMP_ARRAYS_SCHULZ = std::max(
        MAX_TEMP_ARRAYS_SCHULZ, CURR_NUM_TEMP_ARRAYS_SCHULZ);

    // Make it input of Y_n-1
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        std::copy_n(qso->out, qso->N, temp[i].get());
        qso->in = temp[i].get();
    }

    // Get Y . Z . v1
    multiplyNewtonSchulzY(n - 1, s);
    temp.clear();
    --CURR_NUM_TEMP_ARRAYS_SCHULZ;

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        qso->in = init_ins[i];
        for (int j = 0; j < qso->N; ++j)
            qso->out[j] = 1.5 * cur_res[i][j] - 0.5 * qso->out[j];
    }
    --CURR_NUM_TEMP_ARRAYS_SCHULZ;
}


void Qu3DEstimator::multiplyNewtonSchulzZ(int n, double s) {
    if (n == 0) {
        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars)
            std::copy_n(qso->in, qso->N, qso->out);
        return;
    }
    else if (n == 1) {
        multiplyAsVector(0, s);
        ++NUMBER_SCHULZ_MULTIS;

        #pragma omp parallel for schedule(dynamic, 4)
        for (auto &qso : quasars)
            for (int j = 0; j < qso->N; ++j)
                qso->out[j] = 1.5 * qso->in[j] - 0.5 * qso->out[j];
        return;
    }

    // need temp storage
    std::vector<std::unique_ptr<double[]>> temp, cur_res;
    std::vector<double*> init_ins;
    cur_res.reserve(quasars.size());
    init_ins.reserve(quasars.size());

    for (const auto &qso : quasars) {
        cur_res.push_back(std::make_unique<double[]>(qso->N));
        init_ins.push_back(qso->in);
    }
    ++CURR_NUM_TEMP_ARRAYS_SCHULZ;
    MAX_TEMP_ARRAYS_SCHULZ = std::max(
        MAX_TEMP_ARRAYS_SCHULZ, CURR_NUM_TEMP_ARRAYS_SCHULZ);

    // Get z1 to *out
    multiplyNewtonSchulzZ(n - 1, s);

    // Save it to cur_res. Make it input of Z_n-1
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        std::copy_n(qso->out, qso->N, cur_res[i].get());
        qso->in = cur_res[i].get();
    }

    // Get Y . z1 to *out
    multiplyNewtonSchulzY(n - 1, s);

    temp.reserve(quasars.size());
    for (const auto &qso : quasars)
        temp.push_back(std::make_unique<double[]>(qso->N));
    ++CURR_NUM_TEMP_ARRAYS_SCHULZ;
    MAX_TEMP_ARRAYS_SCHULZ = std::max(
        MAX_TEMP_ARRAYS_SCHULZ, CURR_NUM_TEMP_ARRAYS_SCHULZ);

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
    --CURR_NUM_TEMP_ARRAYS_SCHULZ;

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < quasars.size(); ++i) {
        auto &qso = quasars[i];
        qso->in = init_ins[i];
        for (int j = 0; j < qso->N; ++j)
            qso->out[j] = 1.5 * cur_res[i][j] - 0.5 * qso->out[j];
    }
    --CURR_NUM_TEMP_ARRAYS_SCHULZ;
}


void Qu3DEstimator::multiplyCovSmallSqrtNewtonSchulz(int order) {
    static double max_eval = findMaxDiagonalAs(); // estimateMaxEvalAs();
    CURR_NUM_TEMP_ARRAYS_SCHULZ = 0;
    NUMBER_SCHULZ_MULTIS = 0;
    multiplyNewtonSchulzY(order, max_eval);

    // LOG::LOGGER.STD(
    //     "Maximum number of temp arrays %d. Number of multiplications %d.\n",
    //     MAX_TEMP_ARRAYS_SCHULZ, NUMBER_SCHULZ_MULTIS);

    double s = sqrt(max_eval);
    #pragma omp parallel for schedule(dynamic, 4)
    for (auto &qso : quasars)
        for (int i = 0; i < qso->N; ++i)
            qso->truth[i] = qso->out[i] * s;
}
