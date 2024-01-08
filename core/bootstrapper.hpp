#ifndef BOOTSTRAPPER_H
#define BOOTSTRAPPER_H

#include <memory>
#include <random>
#include <string>
#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/one_qso_estimate.hpp"
#include "core/global_numbers.hpp"
#include "core/progress.hpp"
#include "mathtools/matrix_helper.hpp"
#include "mathtools/stats.hpp"


class PoissonRNG {
public:
    PoissonRNG(unsigned long int seed) {
        rng_engine.seed(seed);
    }

    unsigned int generate() {
        return p_dist(rng_engine);
    }

    void fillVector(std::vector<double> &v) {
        for (auto it = v.begin(); it != v.end(); ++it)
            (*it) = generate();
    }

private:
    std::mt19937_64 rng_engine;
    std::poisson_distribution<unsigned int> p_dist;
};


namespace mytime
{
    static double time_spent_on_oneboot_loop = 0, time_spent_on_oneboot_mpi = 0,
                  time_spent_on_oneboot_solve = 0;

    void printfBootstrapTimeSpentDetails()
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
        pgenerator = std::make_unique<PoissonRNG>(process::this_pe);

        if (specifics::FAST_BOOTSTRAP) {
            temppower.resize(nboots * bins::TOTAL_KZ_BINS);
            pcoeff.resize(nboots);
        } else
            temppower.resize(bins::TOTAL_KZ_BINS);

        tempfisher.resize(bins::FISHER_SIZE);
        if (process::this_pe == 0)
            allpowers.resize(nboots * bins::TOTAL_KZ_BINS);

        outlier.resize(nboots, false);
    }
    ~PoissonBootstrapper() { process::updateMemory(getMinMemUsage()); }

    void run(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        // Get thetas to prepare.
        for (auto one_qso = local_queue.begin(); one_qso != local_queue.end(); ++one_qso)
            (*one_qso)->collapseBootstrap();

        LOG::LOGGER.STD("Generating %u bootstrap realizations.\n", nboots);

        if (specifics::FAST_BOOTSTRAP) {
            _fastBootstrap(local_queue);
        }
        else {
            Progress prog_tracker(nboots);
            int ii = 0;
            for (unsigned int jj = 0; jj < nboots; ++jj) {
                bool valid = _oneBoot(ii, local_queue);
                ++prog_tracker;
                if (valid) ++ii;
            }

            remaining_boots = ii;
        }

        if (process::this_pe != 0)
            return;

        double success_ratio = (100. * remaining_boots) / nboots;
        LOG::LOGGER.STD(
            "Number of valid bootstraps %d of %d. Ratio %.2f%%\n",
            remaining_boots, nboots, success_ratio);
        if (success_ratio < 90.)
            LOG::LOGGER.ERR("WARNING: Low success ratio!\n");

        mytime::printfBootstrapTimeSpentDetails();

        _meanBootstrap();
        _medianBootstrap();
    }

private:
    unsigned int nboots, remaining_boots;
    double *invfisher;
    std::unique_ptr<PoissonRNG> pgenerator;
    std::vector<double> temppower, tempfisher, allpowers, pcoeff;
    std::vector<bool> outlier;

    double getMinMemUsage() {
        double memfull = process::getMemoryMB(nboots * bins::TOTAL_KZ_BINS);
        double needed_mem = (
            process::getMemoryMB(bins::FISHER_SIZE)
            + memfull / process::total_pes
        );

        if (specifics::FAST_BOOTSTRAP)
            needed_mem += memfull + process::getMemoryMB(nboots);
        else
            needed_mem += process::getMemoryMB(bins::TOTAL_KZ_BINS);
        return needed_mem;
    }

    void _fastBootstrap(
            std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double t1 = mytime::timer.getTime(), t2 = 0;
        std::fill(temppower.begin(), temppower.end(), 0);

        Progress prog_tracker(local_queue.size());
        for (const auto &one_qso : local_queue) {
            pgenerator->fillVector(pcoeff);
            one_qso->addBootPowerOnly(nboots, pcoeff.data(), temppower.data());
            ++prog_tracker;
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        #if defined(ENABLE_MPI)
        MPI_Reduce(
            temppower.data(),
            allpowers.data(), temppower.size(),
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        #endif

        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;
    }

    bool _oneBoot(
            int jj, std::vector<std::unique_ptr<OneQSOEstimate>> &local_queue
    ) {
        double t1 = mytime::timer.getTime(), t2;

        std::fill(temppower.begin(), temppower.end(), 0);
        std::fill(tempfisher.begin(), tempfisher.end(), 0);

        for (const auto &one_qso : local_queue) {
            int p = pgenerator->generate();
            if (p == 0)
                continue;

            one_qso->addBoot(p, temppower.data(), tempfisher.data());
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_loop += t2 - t1;

        #if defined(ENABLE_MPI)
        MPI_Reduce(
            temppower.data(),
            allpowers.data() + jj * bins::TOTAL_KZ_BINS,
            bins::TOTAL_KZ_BINS,
            MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD
        );
        if (process::this_pe != 0) {
            MPI_Reduce(
                tempfisher.data(),
                nullptr, bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(
                MPI_IN_PLACE,
                tempfisher.data(), bins::FISHER_SIZE,
                MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        #endif

        bool valid = true;
        t1 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_mpi += t1 - t2;

        if (process::this_pe == 0) {
            try {
                mxhelp::LAPACKE_safeSolveCho(
                    tempfisher.data(), bins::TOTAL_KZ_BINS,
                    allpowers.data() + jj * bins::TOTAL_KZ_BINS);
            } catch (std::exception& e) {
                valid = false;
            }
        }

        t2 = mytime::timer.getTime();
        mytime::time_spent_on_oneboot_solve += t2 - t1;

        #if defined(ENABLE_MPI)
        MPI_Bcast(&valid, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        #endif

        return valid;
    }


    void _sandwichInvFisher() {
        auto m = std::make_unique<double[]>(bins::FISHER_SIZE);
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS,
            0.5, invfisher,
            bins::TOTAL_KZ_BINS, tempfisher.data(), bins::TOTAL_KZ_BINS,
            0, m.get(), bins::TOTAL_KZ_BINS);

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS,
            0.5, m.get(),
            bins::TOTAL_KZ_BINS, invfisher, bins::TOTAL_KZ_BINS,
            0, tempfisher.data(), bins::TOTAL_KZ_BINS);
    }


    void _calcuateMean(std::vector<double> &mean) {
        std::fill(mean.begin(), mean.begin() + bins::TOTAL_KZ_BINS, 0);
        for (unsigned int jj = 0; jj < nboots; ++jj) {
            if (outlier[jj]) continue;

            cblas_daxpy(
                bins::TOTAL_KZ_BINS,
                1, allpowers.data() + jj * bins::TOTAL_KZ_BINS, 1,
                mean.data(), 1);
        }
        cblas_dscal(bins::TOTAL_KZ_BINS, 1. / remaining_boots, mean.data(), 1);
    }


    void _calcuateMedian(std::vector<double> &median) {
        median.resize(bins::TOTAL_KZ_BINS);

        for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i) {
            cblas_dcopy(
                nboots, allpowers.data() + i, bins::TOTAL_KZ_BINS,
                pcoeff.data(), 1);
            median[i] = stats::medianOfUnsortedVector(pcoeff);
        }
    }


    void _calcuateMadCovariance(
            const double *median, std::vector<double> &mad_cov
    ) {
        mad_cov.resize(bins::FISHER_SIZE);

        // Subtract median from allpowers
        for (unsigned int n = 0; n < nboots; ++n)
            cblas_daxpy(
                bins::TOTAL_KZ_BINS, -1, median, 1,
                allpowers.data() + n * bins::TOTAL_KZ_BINS, 1);

        for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i) {
            const double *x = allpowers.data() + i;

            for (int j = i; j < bins::TOTAL_KZ_BINS; ++j) {
                const double *y = allpowers.data() + j;

                mxhelp::vector_multiply(
                    nboots, x, bins::TOTAL_KZ_BINS, y, bins::TOTAL_KZ_BINS,
                    pcoeff.data());

                mad_cov[j + i * bins::TOTAL_KZ_BINS] =
                    2.19810276 * stats::medianOfUnsortedVector(pcoeff);
            }
        }

        mxhelp::copyUpperToLower(mad_cov.data(), bins::TOTAL_KZ_BINS);

        if (specifics::FAST_BOOTSTRAP)
            _sandwichInvFisher();
    }


    void _calcuateCovariance(const double *mean, std::vector<double> &cov) {
        auto v = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

        std::fill(cov.begin(), cov.end(), 0);
        for (unsigned int jj = 0; jj < nboots; ++jj) {
            if (outlier[jj]) continue;

            double *x = allpowers.data() + jj * bins::TOTAL_KZ_BINS;
            for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i)
                v[i] = x[i] - mean[i];

            cblas_dsyr(
                CblasRowMajor, CblasUpper,
                bins::TOTAL_KZ_BINS, 1, v.get(), 1,
                cov.data(), bins::TOTAL_KZ_BINS);
        }

        cblas_dscal(
            bins::FISHER_SIZE, 1. / (remaining_boots - 1),
            cov.data(), 1);

        mxhelp::copyUpperToLower(cov.data(), bins::TOTAL_KZ_BINS);

        if (specifics::FAST_BOOTSTRAP)
            _sandwichInvFisher();
    }


    unsigned int _findOutliers(
            const std::vector<double> &mean, const double *invcov
    ) {
        auto v = std::make_unique<double[]>(bins::TOTAL_KZ_BINS),
             y = std::make_unique<double[]>(bins::TOTAL_KZ_BINS);

        unsigned int new_remains = 0;
        double maxChi2 = 8. * sqrt(2. * bins::TOTAL_KZ_BINS);

        for (unsigned int jj = 0; jj < nboots; ++jj) {
            if (outlier[jj]) continue;

            double *x = allpowers.data() + jj * bins::TOTAL_KZ_BINS;
            for (int i = 0; i < bins::TOTAL_KZ_BINS; ++i)
                v[i] = x[i] - mean[i];

            double chi2 = mxhelp::my_cblas_dsymvdot(
                v.get(), invcov, y.get(), bins::TOTAL_KZ_BINS) / 4
                - bins::TOTAL_KZ_BINS;

            if (fabs(chi2) > maxChi2)
                outlier[jj] = true;
            else
                ++new_remains;
        }

        return new_remains;
    }


    void _medianBootstrap() {
        LOG::LOGGER.STD("Calculating median bootstrap covariance.\n");
        _calcuateMedian(temppower);
        _calcuateMadCovariance(temppower.data(), tempfisher);
        _saveData("median");
    }

    void _meanBootstrap() {
        LOG::LOGGER.STD("Calculating mean bootstrap covariance.\n");
        // Calculate mean power, store into temppower
        _calcuateMean(temppower);
        // Calculate the covariance matrix into tempfisher
        _calcuateCovariance(temppower.data(), tempfisher);
        _saveData("mean");

        for (int n = 0; n < 5; ++n) {
            LOG::LOGGER.STD("  Iteration %d...", n + 1);

            // Find outliers
            unsigned int new_remains = _findOutliers(temppower, tempfisher.data());

            LOG::LOGGER.STD("Removed outliers. Remaining %d.\n", new_remains);
            if (new_remains == remaining_boots)
                break;

            remaining_boots = new_remains;

            _calcuateMean(temppower);
            _calcuateCovariance(temppower.data(), tempfisher);
        }
        
        _saveData("mean_pruned");
    }


    void _saveData(const std::string &t) {
        std::string buffer = 
            process::FNAME_BASE + std::string("_bootstrap_") + t
            + std::string(".txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), temppower.data(),
            1, bins::TOTAL_KZ_BINS);

        buffer = process::FNAME_BASE + std::string("_bootstrap_") + t
                 + std::string("_covariance.txt");
        mxhelp::fprintfMatrix(
            buffer.c_str(), tempfisher.data(),
            bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    }
};

#endif
