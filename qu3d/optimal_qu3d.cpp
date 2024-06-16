#include "qu3d/optimal_qu3d.hpp"
#include "qu3d/cosmology_3d.hpp"

#include "core/global_numbers.hpp"
#include "io/logger.hpp"

std::unique_ptr<fidcosmo::FlatLCDM> cosmo;
std::unique_ptr<fidcosmo::ArinyoP3DModel> p3d_model;


void Qu3DEstimator::_readOneDeltaFile(const std::string &fname) {
    qio::PiccaFile pFile(fname);
    int number_of_spectra = pFile.getNumberSpectra();
    std::vector<std::unique_ptr<qio::QSOFile>> local_quasars;
    local_quasars.reserve(number_of_spectra);

    for (int i = 0; i < number_of_spectra; ++i) {
        std::ostringstream fpath;
        fpath << fname << '[' << i + 1 << ']';

        auto qFile = std::make_unique<qio::QSOFile>(&pFile, i + 1);
        qFile->fname = fpath.str();
        qFile->readParameters();

        if (qFile->snr < specifics::MIN_SNR_CUT)
            continue;

        qFile->readData();

        qFile->maskOutliers();
        qFile->cutBoundary(bins::Z_LOWER_EDGE, bins::Z_UPPER_EDGE);

        std::for_each(
            qFile->wave(), qFile->wave() + qFile->size(), [](double &ld) {
                ld = ld / LYA_REST;
            }
        );
        // Convert to distance
        // Convert to ivar again

        local_quasars.push_back(std::move(qFile));
    }

    pFile.closeFile()

    if (local_quasars.empty())
        return;

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

    num_iterations = config.getInteger("NumberOfIterations");

    mesh.ngrid[0] = config.getInteger("NGRID_X");
    mesh.ngrid[1] = config.getInteger("NGRID_Y");
    mesh.ngrid[2] = config.getInteger("NGRID_Z");

    mesh.length[0] = config.getInteger("LENGTH_X");
    mesh.length[1] = config.getInteger("LENGTH_Y");
    mesh.length[2] = config.getInteger("LENGTH_Z");
    mesh.z0 = config.getInteger("ZSTART");

    cosmo = std::make_unique<fidcosmo::FlatLCDM>(config);
    p3d_model = std::make_unique<fidcosmo::ArinyoP3DModel>(config);

    _readQSOFiles(flist, findir);
    mesh.construct();
}
