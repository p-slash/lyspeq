#include "qu3d/optimal_qu3d.hpp"
#include "qu3d/cosmology_3d.hpp"

#include "core/global_numbers.hpp"
#include "io/logger.hpp"

std::unique_ptr<fidcosmo::FlatLCDM> cosmo;
std::unique_ptr<fidcosmo::ArinyoP3DModel> p3d_model;


void Qu3DEstimator::_readOneDeltaFile(const std::string &fname) {
    qio::PiccaFile pFile(fname);
    int number_of_spectra = pFile.getNumberSpectra();
    std::vector<std::unique_ptr<CosmicQuasar>> local_quasars;
    local_quasars.reserve(number_of_spectra);

    for (int i = 0; i < number_of_spectra; ++i) {
        try {
            local_quasars.push_back(
                std::make_unique<CosmicQuasar>(&pFile, i + 1));
        } catch (std::exception& e) {
            std::ostringstream fpath;
            fpath << fname << '[' << i + 1 << ']';
            LOG::LOGGER.ERR(
                "%s. Filename %s.\n", e.what(), fpath.str().c_str());
        }
    }

    pFile.closeFile();

    if (local_quasars.empty())
        return;

    for (auto &qso : local_quasars)
        qso->setRadialComovingDistance(cosmo.get());

    #pragma omp critical
    {
        quasars.reserve(quasars.size() + local_quasars.size());
        std::move(std::begin(local_quasars), std::end(local_quasars),
                  std::back_inserter(quasars));
        local_quasars.clear();
    }
}

void Qu3DEstimator::_readQSOFiles(
        const std::string &flist, const std::string &findir
) {
    double t1 = mytime::timer.getTime(), t2 = 0;
    std::vector<std::string> filepaths;

    LOG::LOGGER.STD("Read delta files.\n");
    qio::PiccaFile::use_cache = false;

    int number_of_files = ioh::readList(flist.c_str(), filepaths);

    #pragma omp parallel for
    for (auto &fq : filepaths) {
        fq.insert(0, findir);  // Add parent directory to file path
        _readOneDeltaFile(fq);
    }

    int init_num_qsos = quasars.size();

    t2 = mytime::timer.getTime();
    LOG::LOGGER.STD("Reading QSO files took %.2f m.\n", t2 - t1);

    if (quasars.empty())
        throw std::runtime_error("No spectrum in queue. Check files & redshift range.");
}


Qu3DEstimator::Qu3DEstimator(ConfigFile &config) {
    config.addDefaults(qu3d_default_parameters);

    std::string
        flist = config.get("FileNameList"),
        findir = config.get("FileInputDir");

    if (flist.empty())
        throw std::invalid_argument("Must pass FileNameList.");
    if (findir.empty())
        throw std::invalid_argument("Must pass FileInputDir.");

    if (findir.back() != '/')
        findir += '/';

    cosmo = std::make_unique<fidcosmo::FlatLCDM>(config);
    p3d_model = std::make_unique<fidcosmo::ArinyoP3DModel>(config);

    _readQSOFiles(flist, findir);

    num_iterations = config.getInteger("NumberOfIterations");
    tolerance = config.getDouble("ConvergenceTolerance");

    mesh.ngrid[0] = config.getInteger("NGRID_X");
    mesh.ngrid[1] = config.getInteger("NGRID_Y");
    mesh.ngrid[2] = config.getInteger("NGRID_Z");

    mesh.length[0] = config.getInteger("LENGTH_X");
    mesh.length[1] = config.getInteger("LENGTH_Y");
    mesh.length[2] = config.getInteger("LENGTH_Z");
    mesh.z0 = config.getInteger("ZSTART");

    mesh.construct();
}


void Qu3DEstimator::multMeshComp() {
    // Reverse interp
    double coord[3];
    mesh.zero_field_k();
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->getCartesianCoords(i, coord);
            mesh.reverseInterpolate(coord, qso->in[i]);   
        }
    }

    // Convolve power
    mesh.fftX2K();
    #pragma omp parallel for
    for (size_t i = 0; i < mesh.size_complex; ++i) {
        double k, kz;
        mesh.getKKzFromIndex(i, k, kz);
        mesh.field_k[i] *= p3d_model->evaluate(k, kz);
    }
    mesh.fftK2X();

    // Interpolate and Weight by Ivar
    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->getCartesianCoords(i, coord);
            qso->out[i] += qso->ivar[i] * mesh.interpolate(coord);   
        }
    }
}


void Qu3DEstimator::updateY(double residual_norm2) {
    double a_down = 0, alpha = 0;

    #pragma omp parallel for
    for (auto &qso : quasars)
        qso->in = qso->search.get();

    /* Multiply C x search into Cy*/
    multiplyCovVector();

    // get a_down
    #pragma omp parallel for reduction(+:a_down)
    for (auto &qso : quasars)
        a_down += cblas_ddot(qso->N, qso->search.get(), 1, qso->Cy.get(), 1);

    alpha = residual_norm2 / a_down;

    /* Update y in the search direction, restore qso->in
       Update residual*/
    #pragma omp parallel for
    for (auto &qso : quasars) {
        cblas_daxpy(qso->N, alpha, qso->search.get(), 1, qso->y.get(), 1);
        cblas_daxpy(qso->N, -alpha, qso->residual.get(), 1, qso->Cy.get(), 1);
        qso->in = qso->y.get();
    }
}


void Qu3DEstimator::conjugateGradientDescent() {
    multiplyCovVector();

    #pragma omp parallel for
    for (auto &qso : quasars) {
        for (int i = 0; i < qso->N; ++i) {
            qso->residual[i] = qso->Cy[i] - qso->qFile->delta()[i];
            qso->search[i] = qso->residual[i];
        }
    }

    double old_residual_norm2 = calculateResidualNorm2();

    if (sqrt(old_residual_norm2) < tolerance)
        return;

    for (int niter = 0; niter < num_iterations; ++niter) {
        updateY(old_residual_norm2);
        multiplyCovVector();

        double new_residual_norm2 = calculateResidualNorm2();
        if (sqrt(new_residual_norm2) < tolerance)
            return;

        double beta = new_residual_norm2 / old_residual_norm2;
        old_residual_norm2 = new_residual_norm2;
        calculateNewDirection(beta);
    }
}
