#ifndef BOOTSTRAPPER_H
#define BOOTSTRAPPER_H

#include <algorithm>
#include <memory>
#include <random>
#include <string>

#include "core/one_qso_estimate.hpp"
#include "core/omp_manager.hpp"
#include "core/mpi_manager.hpp"
#include "core/global_numbers.hpp"
#include "core/progress.hpp"
#include "mathtools/matrix_helper.hpp"
#include "mathtools/stats.hpp"
#include "io/bootstrap_file.hpp"


class PoissonRNG {
public:
    PoissonRNG(int seed) {
        rng_engine.seed(seed);
    }

    double generate() {
        return p_dist(rng_engine);
    }

    void fillVector(double *v, unsigned int size) {
        for (unsigned int i = 0; i < size; ++i)
            v[i] = generate();
    }

private:
    std::mt19937_64 rng_engine;
    std::poisson_distribution<int> p_dist;
};


namespace mytime
{
    static double time_spent_on_oneboot_loop = 0, time_spent_on_oneboot_mpi = 0,
                  time_spent_on_oneboot_solve = 0;

    inline void printfBootstrapTimeSpentDetails()
    {
        LOG::LOGGER.STD(
            "Total time spent in loop is %.2f mins.\n"
            "Total time spent in MPI is %.2f mins.\n"
            "Total time spent in solve is %.2f mins.\n",
            time_spent_on_oneboot_loop, time_spent_on_oneboot_mpi,
            time_spent_on_oneboot_solve);
    }
}

class PoissonBootstrapper {
public:
    PoissonBootstrapper(int num_boots, double *ifisher)
            : nboots(num_boots), remaining_boots(num_boots), invfisher(ifisher)
    {
        process::updateMemory(-getMinMemUsage());
        pgenerator = std::make_unique<PoissonRNG>(mympi::this_pe);

        if (specifics::FAST_BOOTSTRAP) {
            temppower = std::make_unique<double[]>(nboots * bins::TOTAL_KZ_BINS);
            pcoeff = std::make_unique<double[]>(nboots);
        } else
            temppower = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

        tempfisher = std::make_unique<double[]>(bins::FISHER_SIZE);
        if (mympi::this_pe == 0)
            allpowers = std::make_unique<double[]>(nboots * bins::TOTAL_KZ_BINS);

        // outlier = std::make_unique<bool[]>(nboots);
    }

    PoissonBootstrapper(const std::string &fname) {
        if (mympi::this_pe != 0)
            return;

        if (mympi::total_pes > 1)
            LOG::LOGGER.ERR("No need for MPI.\n");

        ioh::readBootstrapRealizations(
            fname, allpowers, slvF,
            nboots, bins::NUMBER_OF_K_BANDS, bins::NUMBER_OF_Z_BINS,
            specifics::FAST_BOOTSTRAP
        );

        remaining_boots = nboots;
        bins::TOTAL_KZ_BINS = bins::NUMBER_OF_K_BANDS * bins::NUMBER_OF_Z_BINS;
        bins::FISHER_SIZE = bins::TOTAL_KZ_BINS * bins::TOTAL_KZ_BINS;

        if (myomp::getMaxNumThreads() > bins::TOTAL_KZ_BINS) {
            LOG::LOGGER.STD("Using only %d OMP threads.", bins::TOTAL_KZ_BINS);
            myomp::setNumThreads(bins::TOTAL_KZ_BINS);
        }

        pcoeff = std::make_unique<double[]>(nboots * bins::TOTAL_KZ_BINS);
        tempfisher = std::make_unique<double[]>(bins::FISHER_SIZE);
        temppower = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);
        // outlier = std::make_unique<bool[]>(nboots);

        invfisher = slvF.get();
    }

    ~PoissonBootstrapper() { process::updateMemory(getMinMemUsage()); }

    void run() { _meanBootstrap(); _medianBootstrap(); }

    void run(
            const std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        LOG::LOGGER.STD("Generating %u bootstrap realizations.\n", nboots);

        std::string comment;
        if (specifics::FAST_BOOTSTRAP) {
            LOG::LOGGER.STD("Using FastBootstrap method.\n");
            comment = "FastBootstrap method. Realizations are not normalized. "
                      "Apply y = 0.5 gemv(SOLV_INVF, x)";
            _fastBootstrap(local_queue);
        }
        else {
            comment = "Complete bootstrap method. Realizations are normalized.";
            Progress prog_tracker(nboots);
            int ii = 0;
            for (unsigned int jj = 0; jj < nboots; ++jj) {
                bool valid = _oneBoot(ii, local_queue);
                ++prog_tracker;
                if (valid) ++ii;
            }

            remaining_boots = ii;
        }

        if (mympi::this_pe != 0)
            return;

        double success_ratio = (100. * remaining_boots) / nboots;
        LOG::LOGGER.STD(
            "Number of valid bootstraps %d of %d. Ratio %.2f%%\n",
            remaining_boots, nboots, success_ratio);
        if (success_ratio < 90.)
            LOG::LOGGER.ERR("WARNING: Low success ratio!\n");

        mytime::printfBootstrapTimeSpentDetails();

        if (specifics::SAVE_BOOTREALIZATIONS) {
            std::string out_fname = ioh::saveBootstrapRealizations(
                process::FNAME_BASE, allpowers.get(), invfisher,
                nboots, bins::NUMBER_OF_K_BANDS, bins::NUMBER_OF_Z_BINS,
                specifics::FAST_BOOTSTRAP, comment.c_str());
            LOG::LOGGER.STD(
                "Bootstrap realizations saved as %s.\n",
                out_fname.c_str() + 1);
        }

        // Median bootstrap is too time consuming.
        _meanBootstrap();
    }

private:
    unsigned int nboots, remaining_boots;
    double *invfisher;
    std::unique_ptr<PoissonRNG> pgenerator;
    std::unique_ptr<double[]> temppower, tempfisher, allpowers, pcoeff, slvF;
    // std::unique_ptr<bool[]> outlier;

    double getMinMemUsage() {
        double memfull = process::getMemoryMB(nboots * bins::TOTAL_KZ_BINS);
        double needed_mem = (
            process::getMemoryMB(bins::FISHER_SIZE)
            + memfull / mympi::total_pes
        );

        if (specifics::FAST_BOOTSTRAP)
            needed_mem += memfull + process::getMemoryMB(nboots);
        else
            needed_mem += process::getMemoryMB(bins::TOTAL_KZ_BINS);
        return needed_mem;
    }

    void _fastBootstrap(
            const std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        /* After this function call,
            - On main task: pcoeff and temppower memories are swapped, such that
            pcoeff is size of nboot * Nkz and temppower is size of Nkz
            - On other tasks, pcoeff and temppower are released.
        */
        double t1 = mytime::timer.getTime(), t2 = 0;
        // std::fill_n(temppower.get(), nboots * bins::TOTAL_KZ_BINS, 0);

        Progress prog_tracker(local_queue.size());
        for (const auto &one_qso : local_queue) {
            pgenerator->fillVector(pcoeff.get(), nboots);
            one_qso->addBootPowerOnly(nboots, pcoeff.get(), temppower.get());
            ++prog_tracker;
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        mympi::reduceToOther(
            temppower.get(), allpowers.get(), nboots * bins::TOTAL_KZ_BINS);

        if (mympi::this_pe == 0) {
            temppower.swap(pcoeff);
        } else {
            temppower.reset();
            pcoeff.reset();
        }

        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;
    }

    bool _oneBoot(
            int jj, const std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double t1 = mytime::timer.getTime(), t2;

        std::fill_n(temppower.get(), bins::TOTAL_KZ_BINS, 0);
        std::fill_n(tempfisher.get(), bins::FISHER_SIZE, 0);

        for (const auto &one_qso : local_queue) {
            int p = pgenerator->generate();
            if (p == 0)
                continue;

            one_qso->addBoot(p, temppower.get(), tempfisher.get());
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        mympi::reduceToOther(
            temppower.get(), allpowers.get() + jj * bins::TOTAL_KZ_BINS,
            bins::TOTAL_KZ_BINS);
        mympi::reduceInplace(tempfisher.get(), bins::FISHER_SIZE);

        bool valid = true;
        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;

        if (mympi::this_pe == 0) {
            try {
                mxhelp::LAPACKE_safeSolveCho(
                    tempfisher.get(), bins::TOTAL_KZ_BINS,
                    allpowers.get() + jj * bins::TOTAL_KZ_BINS);
            } catch (std::exception& e) {
                valid = false;
            }
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_solve += t2 - t1;
        mympi::bcast(&valid);

        return valid;
    }


    void _sandwichInvFisher() {
        auto m = std::make_unique<double[]>(bins::FISHER_SIZE);
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS, 1,
            invfisher, bins::TOTAL_KZ_BINS,
            tempfisher.get(), bins::TOTAL_KZ_BINS,
            0, m.get(), bins::TOTAL_KZ_BINS);

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS, 1,
            m.get(), bins::TOTAL_KZ_BINS,
            invfisher, bins::TOTAL_KZ_BINS,
            0, tempfisher.get(), bins::TOTAL_KZ_BINS);
    }


    void _calcuateMean(double *mean) {
        double t1 = mytime::timer.getTime(), t2 = 0;

        std::fill_n(mean, bins::TOTAL_KZ_BINS, 0);
        for (unsigned int jj = 0; jj < nboots; ++jj) {
            // if (outlier[jj]) continue;

            cblas_daxpy(
                bins::TOTAL_KZ_BINS,
                1, allpowers.get() + jj * bins::TOTAL_KZ_BINS, 1,
                mean, 1);
        }
        cblas_dscal(bins::TOTAL_KZ_BINS, 1. / remaining_boots, mean, 1);

        t2 = mytime::timer.getTime();
        LOG::LOGGER.STD("  Total time spent in mean is %.2f mins, ", t2 - t1);
    }


    void _calcuateMedian(double *median) {
        double t1 = mytime::timer.getTime(), t2 = 0;

        #pragma omp parallel for
        for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i) {
            double *buf = pcoeff.get() + nboots * myomp::getThreadNum();

            std::copy_n(allpowers.get() + i * nboots, nboots, buf);
            median[i] = stats::medianOfUnsortedVector(buf, nboots);
        }

        t2 = mytime::timer.getTime();
        LOG::LOGGER.STD("median is %.2f mins, ", t2 - t1);
    }


    void _calcuateMadCovariance(const double *median, double *mad_cov) {
        double t1 = mytime::timer.getTime(), t2 = 0;

        // Subtract median from allpowers
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i)
            for (unsigned int n = 0; n < nboots; ++n)
                allpowers[n + i * nboots] -= median[i];

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i) {
            for (int j = i; j < bins::TOTAL_KZ_BINS; ++j) {
                double *buf = pcoeff.get() + nboots * myomp::getThreadNum();

                mxhelp::vector_multiply(
                    nboots, allpowers.get() + i * nboots,
                    allpowers.get() + j * nboots, buf);

                mad_cov[j + i * bins::TOTAL_KZ_BINS] =
                    stats::medianOfUnsortedVector(buf, nboots);
                // 2.19810276
            }
        }

        mxhelp::copyUpperToLower(mad_cov, bins::TOTAL_KZ_BINS);

        t2 = mytime::timer.getTime();
        LOG::LOGGER.STD("MAD covariance is %.2f mins.\n", t2 - t1);
    }


    void _calcuateCovariance(const double *mean, double *cov) {
        double t1 = mytime::timer.getTime(), t2 = 0;
        auto v = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

        std::fill_n(cov, bins::FISHER_SIZE, 0);
        for (unsigned int jj = 0; jj < nboots; ++jj) {
            // if (outlier[jj]) continue;

            double *x = allpowers.get() + jj * bins::TOTAL_KZ_BINS;
            for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i)
                v[i] = x[i] - mean[i];

            cblas_dsyr(
                CblasRowMajor, CblasUpper,
                bins::TOTAL_KZ_BINS, 1, v.get(), 1,
                cov, bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(
            bins::FISHER_SIZE, 1. / (remaining_boots - 1),
            cov, 1);

        mxhelp::copyUpperToLower(cov, bins::TOTAL_KZ_BINS);

        t2 = mytime::timer.getTime();
        LOG::LOGGER.STD("covariance is %.2f mins.\n", t2 - t1);
    }

    void _medianBootstrap() {
        LOG::LOGGER.STD("Calculating median bootstrap covariance.\n");

        double t1 = mytime::timer.getTime(), t2 = 0;
        mxhelp::transpose_copy(
            allpowers.get(), pcoeff.get(), nboots, bins::TOTAL_KZ_BINS);
        allpowers.swap(pcoeff);
        t2 = mytime::timer.getTime();
        LOG::LOGGER.STD("  Total time spent in transpose_copy is %.2f mins, ", t2 - t1);

        _calcuateMedian(temppower.get());
        _calcuateMadCovariance(temppower.get(), tempfisher.get());

        _saveData("median");
    }

    void _meanBootstrap() {
        LOG::LOGGER.STD("Calculating mean bootstrap covariance.\n");
        // Calculate mean power, store into temppower
        _calcuateMean(temppower.get());
        // Calculate the covariance matrix into tempfisher
        _calcuateCovariance(temppower.get(), tempfisher.get());
        _saveData("mean");

        /* Find outliers not useful.
        for (int n = 0; n < 5; ++n) {
            LOG::LOGGER.STD("  Iteration %d...", n + 1);

            // Find outliers
            unsigned int new_remains = _findOutliers(temppower.get(), tempfisher.get());

            LOG::LOGGER.STD("Removed outliers. Remaining %d.\n  ", new_remains);
            if (new_remains == remaining_boots)
                break;

            remaining_boots = new_remains;

            _calcuateMean(temppower.get());
            _calcuateCovariance(temppower.get(), tempfisher.get());
        }
        
        _saveData("mean_pruned"); */
    }


    void _saveData(const std::string &t) {
        std::string buffer = 
            process::FNAME_BASE + std::string("_bootstrap_") + t
            + std::string(".txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), temppower.get(),
            1, bins::TOTAL_KZ_BINS);

        if (specifics::FAST_BOOTSTRAP) {
            buffer = process::FNAME_BASE + std::string("_bootstrap_") + t
                     + std::string("_fisher_matrix.txt");

            cblas_dscal(bins::FISHER_SIZE, 0.25, tempfisher.get(), 1);
            mxhelp::fprintfMatrix(
                buffer.c_str(), tempfisher.get(),
                bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

            _sandwichInvFisher();
        }

        buffer = process::FNAME_BASE + std::string("_bootstrap_") + t
                 + std::string("_covariance.txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), tempfisher.get(),
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }

    /* Find outliers not useful.
    unsigned int _findOutliers(
            const double *mean, const double *invcov
    ) {
        double t1 = mytime::timer.getTime(), t2 = 0;
        auto v = std::make_unique<double[]>(bins::TOTAL_KZ_BINS),
             y = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

        unsigned int new_remains = 0;
        double maxChi2 = 8. * sqrt(2. * bins::TOTAL_KZ_BINS);

        for (unsigned int jj = 0; jj < nboots; ++jj) {
            double *x = allpowers.get() + jj * bins::TOTAL_KZ_BINS;
            for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i)
                v[i] = x[i] - mean[i];

            pcoeff[jj] = mxhelp::my_cblas_dsymvdot(
                v.get(), invcov, y.get(), bins::TOTAL_KZ_BINS) / 4
                - bins::TOTAL_KZ_BINS;

            outlier[jj] = fabs(pcoeff[jj]) > maxChi2;
            if (!outlier[jj])  ++new_remains;
        }

        auto chi2s = stats::getCdfs(pcoeff.get(), nboots);
        LOG::LOGGER.STD("Max chi2: %.3e :: Chi2s:", maxChi2);
        for (auto i = chi2s.cbegin(); i != chi2s.cend(); ++i)
            LOG::LOGGER.STD("   %.3e", *i);

        t2 = mytime::timer.getTime();
        LOG::LOGGER.STD(
            "\n    Time spent in finding outliers is %.2f mins. ", t2 - t1);
        return new_remains;
    } */
};

#endif
