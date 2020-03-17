#include "core/quadratic_estimate.hpp"

#include <cmath>
#include <algorithm> // std::for_each
#include <numeric>
#include <memory> // std::default_delete
#include <cstdio>
#include <cstdlib> // system
#include <cstring> // strcpy
#include <cassert>
#include <stdexcept>
#include <string>
#include <sstream>      // std::ostringstream

#include <gsl/gsl_cblas.h>

#if defined(ENABLE_MPI)
#include "mpi.h" 
#endif

#include "core/matrix_helper.hpp"
#include "core/global_numbers.hpp"
#include "core/fiducial_cosmology.hpp"
#include "io/io_helper_functions.hpp"
#include "io/logger.hpp"

//-------------------------------------------------------
#define MPI_FNA_TAG 1
#define MPI_CPU_TAG 0
#define MPI_VEC_TAG 2

void mpi_send_string(int target_pe, std::string &str)
{
    int length = str.length()+1;
    MPI_Send(str.c_str(), length, MPI_CHAR, target_pe, MPI_FNA_TAG, MPI_COMM_WORLD);
}

void mpi_recv_string(int source_pe, std::string &str)
{
    // First recieve the size of the transmission
    int length;
    MPI_Status status;
    MPI_Probe(source_pe, MPI_FNA_TAG, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_CHAR, &length);

    char *buf = new char[length+1];
    MPI_Recv(buf, length+1, MPI_CHAR, source_pe, MPI_FNA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    str = std::string(buf);

    delete [] buf;
}

void mpi_bcast_string(int root_pe, std::string &str)
{
    int length = str.length()+1;
    MPI_Bcast(&length, 1, MPI_INT, root_pe, MPI_COMM_WORLD);

    char *buf = new char[length];
    
    if (process::this_pe == root_pe)
        strcpy(buf, str.c_str());

    MPI_Bcast(buf, length, MPI_CHAR, root_pe, MPI_COMM_WORLD);

    if (process::this_pe != root_pe)
        str = std::string(buf);
    
    delete [] buf;
}

void mpi_send_vec(int target_pe, std::vector<std::pair<double, std::string>> &vec)
{
    int size = vec.size();
    MPI_Send(&size, 1, MPI_INT, target_pe, MPI_VEC_TAG, MPI_COMM_WORLD);
    for (std::vector<std::pair<double, std::string>>::iterator it = vec.begin(); it != vec.end(); ++it)
    {
        MPI_Send(&(it->first), 1, MPI_DOUBLE, target_pe, MPI_VEC_TAG, MPI_COMM_WORLD);
        mpi_send_string(target_pe, it->second);
    }
}

void mpi_recv_vec(int source_pe, std::vector<std::pair<double, std::string>> &vec)
{
    int size;
    MPI_Recv(&size, 1, MPI_INT, source_pe, MPI_VEC_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vec.reserve(size);

    for (int i = 0; i < size; ++i)
    {
        double buf_cpu;
        std::string buf;

        MPI_Recv(&buf_cpu, 1, MPI_DOUBLE, source_pe, MPI_VEC_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mpi_recv_string(source_pe, buf);
        vec.push_back(std::make_pair(buf_cpu, buf));
    }
}

void mpi_bcast_cpu_fname_vec(std::vector<std::pair<double, std::string>> &cpu_fname_vec, int root_pe=0)
{
    int size;

    // Root send the size of the vector
    if (process::this_pe == root_pe)
        size = cpu_fname_vec.size();
    
    MPI_Bcast(&size, 1, MPI_INT, root_pe, MPI_COMM_WORLD);
    
    if (process::this_pe != 0)
    {
        cpu_fname_vec.clear();
        cpu_fname_vec.reserve(size);
    }

    for (int i = 0; i < size; ++i)
    {
        double cpu_time   = cpu_fname_vec[i].first;
        std::string fname = cpu_fname_vec[i].second;

        MPI_Bcast(&cpu_time, 1, MPI_DOUBLE, root_pe, MPI_COMM_WORLD);
        mpi_bcast_string(root_pe, fname);

        if (process::this_pe != root_pe)
            cpu_fname_vec.push_back(std::make_pair(cpu_time, fname));
    }
}

void mpi_merge_sorted_arrays(int height, int Npe, int id, 
    std::vector<std::pair<double, std::string>> &local_cpu_fname_vec)
{
    if (Npe == 1)  // We have reached to the final
        return;

    int parent_pe, child_pe, next_Npe, local_size = local_cpu_fname_vec.size();
    next_Npe = (Npe + 1) / 2;

    // Given a height, parent PEs are 2**(height+1), 1 << (height+1), height starts from 0 at the bottom (all PEs)
    // This means e.g. at height 0 we need to map: 3->2, 2->2 and 5->4, 4->4 to find parents
    parent_pe = (id & ~(1 << height));
    
    if (id == parent_pe)
    {
        // If this is the parent PE, receive from the child.
        child_pe = (id | (1 << height));

        // If childless, carry on to the next cycle
        if (child_pe >= process::total_pes)
        {
            mpi_merge_sorted_arrays(height+1, next_Npe, id, local_cpu_fname_vec);
            return;
        }

        // Receive child's sorted vector
        std::vector<std::pair<double, std::string>> childs_sorted_vec;
        mpi_recv_vec(child_pe, childs_sorted_vec);
        
        // Merge sorted these two arrays
        local_cpu_fname_vec.insert(local_cpu_fname_vec.end(), childs_sorted_vec.begin(), childs_sorted_vec.end());
        std::inplace_merge(local_cpu_fname_vec.begin(), local_cpu_fname_vec.begin()+local_size, 
            local_cpu_fname_vec.end());
       
        // Recursive call 
        mpi_merge_sorted_arrays(height+1, next_Npe, id, local_cpu_fname_vec);
    }
    else
    {
        // Child PE sends to its parent
        mpi_send_vec(parent_pe, local_cpu_fname_vec);
    }
}

// int index_of_min_element(double *a, int size)
// {
//     int i = 0;
//     for (int j = 1; j < size; ++j)
//     {
//         if (a[j] < a[i])    i = j;
//     }

//     return i;
// }
//-------------------------------------------------------


OneDQuadraticPowerEstimate::OneDQuadraticPowerEstimate(const char *fname_list, const char *dir)
{
    Z_BIN_COUNTS = new int[bins::NUMBER_OF_Z_BINS+2]();

    // Allocate memory
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        dbt_estimate_sum_before_fisher_vector[dbt_i] = new double[bins::TOTAL_KZ_BINS];
        dbt_estimate_fisher_weighted_vector[dbt_i]   = new double[bins::TOTAL_KZ_BINS];
    }

    previous_power_estimate_vector = new double[bins::TOTAL_KZ_BINS];
    current_power_estimate_vector  = new double[bins::TOTAL_KZ_BINS]();
    fisher_matrix_sum              = new double[bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS];
    inverse_fisher_matrix_sum      = new double[bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS];

    isFisherInverted = false; 

    _readQSOFiles(fname_list, dir);
}

void OneDQuadraticPowerEstimate::_readQSOFiles(const char *fname_list, const char *dir)
{
    double cpu_t_temp, t1, t2;

    LOG::LOGGER.STD("Initial reading of quasar spectra and estimating CPU time.\n");

    std::vector<std::string> fpaths;
    NUMBER_OF_QSOS = ioh::readList(fname_list, fpaths);
    
    // Each PE reads a different section of files
    // They sort individually, then merge in pairs
    // Finally, the master PE broadcasts the sorted full array
    int delta_nfiles = NUMBER_OF_QSOS / process::total_pes,
        fstart_this  = delta_nfiles * process::this_pe,
        fend_this    = delta_nfiles * (process::this_pe + 1);
    
    if (process::this_pe == process::total_pes - 1)
        fend_this = NUMBER_OF_QSOS;

    t2 = mytime::getTime();

    LOG::LOGGER.STD("Rotated the file list so that each PE accesses different files at a given time. "
        "It took %.2f m.\n", t2-t1);

    // Create vector for QSOs & read
    cpu_fname_vector.reserve(NUMBER_OF_QSOS);

    for (std::vector<std::string>::iterator fq = fpaths.begin()+fstart_this; fq != fpaths.begin()+fend_this; ++fq)
    {
        fq->insert(0, "/");
        fq->insert(0, dir);

        OneQSOEstimate q_temp(*fq);
        cpu_t_temp = q_temp.getComputeTimeEst();
        
        ++Z_BIN_COUNTS[q_temp.ZBIN + 1];

        if (cpu_t_temp != 0)
            cpu_fname_vector.push_back(std::make_pair(cpu_t_temp, *fq));
    }
    
    // Print out time it took to read all files into vector
    t1 = mytime::getTime();
    LOG::LOGGER.STD("Reading QSO files took %.2f m.\n", t1-t2);
    
    // MPI Reduce ZBIN_COUNTS
    MPI_Allreduce(MPI_IN_PLACE, Z_BIN_COUNTS, bins::NUMBER_OF_Z_BINS+2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    NUMBER_OF_QSOS_OUT = Z_BIN_COUNTS[0] + Z_BIN_COUNTS[bins::NUMBER_OF_Z_BINS+1];

    LOG::LOGGER.STD("Z bin counts: ");
    for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS+2; zm++)
        LOG::LOGGER.STD("%d ", Z_BIN_COUNTS[zm]);
    LOG::LOGGER.STD("\nNumber of quasars: %d\nQSOs in z bins: %d\n", NUMBER_OF_QSOS, NUMBER_OF_QSOS - NUMBER_OF_QSOS_OUT);

    LOG::LOGGER.STD("Sorting with respect to estimated cpu time.\n");
    std::sort(cpu_fname_vector.begin(), cpu_fname_vector.end()); // Ascending order
    mpi_merge_sorted_arrays(0, process::total_pes, process::this_pe, cpu_fname_vector);
    mpi_bcast_cpu_fname_vec(cpu_fname_vector);

    // Print out time it took to sort files wrt CPU time
    t2 = mytime::getTime();
    LOG::LOGGER.STD("Sorting took %.2f m.\n", t2-t1);
}

OneDQuadraticPowerEstimate::~OneDQuadraticPowerEstimate()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        delete [] dbt_estimate_sum_before_fisher_vector[dbt_i];
        delete [] dbt_estimate_fisher_weighted_vector[dbt_i];
    }

    delete [] previous_power_estimate_vector;
    delete [] current_power_estimate_vector;

    delete [] fisher_matrix_sum;
    delete [] inverse_fisher_matrix_sum;
}

void OneDQuadraticPowerEstimate::invertTotalFisherMatrix()
{
    double t = mytime::getTime();

    LOG::LOGGER.STD("Inverting Fisher matrix.\n");
    
    gsl_matrix_view fisher_mv    = gsl_matrix_view_array(fisher_matrix_sum, 
        bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    gsl_matrix_view invfisher_mv = gsl_matrix_view_array(inverse_fisher_matrix_sum, 
        bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

    gsl_matrix *fisher_copy = gsl_matrix_alloc(bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    gsl_matrix_memcpy(fisher_copy, &fisher_mv.matrix);
    
    mxhelp::invertMatrixLU(fisher_copy, &invfisher_mv.matrix);
    
    gsl_matrix_free(fisher_copy);

    isFisherInverted = true;

    t = mytime::getTime() - t;
    mytime::time_spent_on_f_inv += t;
}

void OneDQuadraticPowerEstimate::computePowerSpectrumEstimates()
{
    assert(isFisherInverted);

    LOG::LOGGER.STD("Estimating power spectrum.\n");

    std::copy(current_power_estimate_vector, current_power_estimate_vector + bins::TOTAL_KZ_BINS, 
        previous_power_estimate_vector);

    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
    {
        cblas_dsymv(CblasRowMajor, CblasUpper,
                    bins::TOTAL_KZ_BINS, 0.5, inverse_fisher_matrix_sum, bins::TOTAL_KZ_BINS,
                    dbt_estimate_sum_before_fisher_vector[dbt_i], 1,
                    0, dbt_estimate_fisher_weighted_vector[dbt_i], 1);
    }

    std::copy(dbt_estimate_fisher_weighted_vector[0], dbt_estimate_fisher_weighted_vector[0] + bins::TOTAL_KZ_BINS, 
        current_power_estimate_vector);
    mxhelp::vector_sub(current_power_estimate_vector, dbt_estimate_fisher_weighted_vector[1], bins::TOTAL_KZ_BINS);
    mxhelp::vector_sub(current_power_estimate_vector, dbt_estimate_fisher_weighted_vector[2], bins::TOTAL_KZ_BINS);
}

void OneDQuadraticPowerEstimate::_readScriptOutput(double *script_power, const char *fname, void *itsfits)
{
    int fr;
    FILE *tmp_fit_file = ioh::open_file(fname, "r");

    if (itsfits != NULL)
    {
        fidpd13::pd13_fit_params *iteration_fits = (fidpd13::pd13_fit_params *) itsfits;

        fr = fscanf(tmp_fit_file, "%le %le %le %le %le %le\n",
                &iteration_fits->A, &iteration_fits->n, &iteration_fits->alpha,
                &iteration_fits->B, &iteration_fits->beta, &iteration_fits->lambda);

        if (fr != 6)
            throw std::runtime_error("Reading fit parameters from tmp_fit_file!");
    }
    
    int kn, zm;
    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; i_kz++)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        fr = fscanf(tmp_fit_file, "%le\n", &script_power[i_kz]);

        if (fr != 1)
            throw std::runtime_error("Reading fit power values from tmp_fit_file!");

        script_power[i_kz] -= powerSpectrumFiducial(kn, zm);
    }

    fclose(tmp_fit_file);
}

// Note that fitting is done deviations plus fiducial power
void OneDQuadraticPowerEstimate::_fitPowerSpectra(double *fitted_power)
{
    static fidpd13::pd13_fit_params iteration_fits = fidpd13::FIDUCIAL_PD13_PARAMS;
    
    char tmp_ps_fname[320], tmp_fit_fname[320];
    ioh::create_tmp_file(tmp_ps_fname, process::TMP_FOLDER);
    ioh::create_tmp_file(tmp_fit_fname, process::TMP_FOLDER);
    
    writeSpectrumEstimates(tmp_ps_fname);

    std::ostringstream command;
    command << "lorentzian_fit.py " << tmp_ps_fname << " " << tmp_fit_fname << " "
            << iteration_fits.A << " " << iteration_fits.n << " " << iteration_fits.n << " ";

    // Do not pass redshift parameters if there is only one redshift bin
    if (bins::NUMBER_OF_Z_BINS > 1)  command << iteration_fits.B << " " << iteration_fits.beta << " ";
    
    command << iteration_fits.lambda;
    
    if (process::this_pe == 0) 
    {
        command
            << " >> " << LOG::LOGGER.getFileName(LOG::TYPE::STD);   
    }

    LOG::LOGGER.STD("%s\n", command.str().c_str());
    LOG::LOGGER.close();

    // Print from python does not go into LOG::LOGGER
    int s1 = system(command.str().c_str());
    
    LOG::LOGGER.reopen();
    remove(tmp_ps_fname);

    if (s1 != 0)
    {
        LOG::LOGGER.ERR("Error in fitting.\n");  
        throw std::runtime_error("fitting error");
    }

    _readScriptOutput(fitted_power, tmp_fit_fname, &iteration_fits);
    remove(tmp_fit_fname);
}

void OneDQuadraticPowerEstimate::_smoothPowerSpectra(double *smoothed_power)
{
    char tmp_ps_fname[320], tmp_smooth_fname[320];
    
    ioh::create_tmp_file(tmp_ps_fname, process::TMP_FOLDER);
    ioh::create_tmp_file(tmp_smooth_fname, process::TMP_FOLDER);

    writeSpectrumEstimates(tmp_ps_fname);

    std::ostringstream command;
    command << "smbivspline.py " << tmp_ps_fname << " " << tmp_smooth_fname; 
    
    if (process::this_pe == 0) 
    {
        command
            << " >> " << LOG::LOGGER.getFileName(LOG::TYPE::STD);   
    }
      
    LOG::LOGGER.STD("%s\n", command.str().c_str());
    LOG::LOGGER.close();

    // Print from python does not go into LOG::LOGGER
    int s1 = system(command.str().c_str());
    
    LOG::LOGGER.reopen();
    remove(tmp_ps_fname);

    if (s1 != 0)
    {
        LOG::LOGGER.ERR("Error in smoothing.\n");  
        throw std::runtime_error("smoothing error");
    }

    _readScriptOutput(smoothed_power, tmp_smooth_fname);
    remove(tmp_smooth_fname);
}

void OneDQuadraticPowerEstimate::_loadBalancing(std::vector<OneQSOEstimate*> &local_queue)
{
    LOG::LOGGER.STD("Load balancing for %d threads available.\n", process::total_pes);
    
    double load_balance_time = mytime::getTime();
    
    std::vector<double> bucket_time(process::total_pes, 0);

    std::vector<std::pair <double, std::string>>::reverse_iterator qe = cpu_fname_vector.rbegin();
    for (; qe != cpu_fname_vector.rend(); ++qe)
    {
        // find min time bucket
        auto min_bt = std::min_element(bucket_time.begin(), bucket_time.end());
        // add max time consuming to that bucket
        (*min_bt) += qe->first;

        if (std::distance(bucket_time.begin(), min_bt) == process::this_pe)
        {
            // Construct and add queue
            OneQSOEstimate *q_temp = new OneQSOEstimate(qe->second);
            local_queue.push_back(q_temp);
        }
    }

    double ave_balance = std::accumulate(bucket_time.begin(), bucket_time.end(), 0.) / process::total_pes;

    LOG::LOGGER.STD("Off-Balance: ");
    for (std::vector<double>::iterator it = bucket_time.begin(); it != bucket_time.end(); ++it)
        LOG::LOGGER.STD("%.1e ", (*it)/ave_balance-1);
    LOG::LOGGER.STD("\n");

    load_balance_time = mytime::getTime() - load_balance_time;
    
    LOG::LOGGER.STD("Load balancing took %.2f sec.\n", load_balance_time*60.);
}

void OneDQuadraticPowerEstimate::iterate(int number_of_iterations, const char *fname_base)
{
    double total_time = 0, total_time_1it = 0;
    double *powerspectra_fits = new double[bins::TOTAL_KZ_BINS]();

    std::vector<OneQSOEstimate*> local_queue;
    _loadBalancing(local_queue);

    for (int i = 0; i < number_of_iterations; i++)
    {
        LOG::LOGGER.STD("Iteration number %d of %d.\n", i+1, number_of_iterations);
        
        total_time_1it = mytime::getTime();
    
        // Set total Fisher matrix and omn before F to zero for all k, z bins
        initializeIteration();

        #if 0
        double thread_time = mytime::getTime();

        LOG::LOGGER.STD("Start working in %d/%d thread with %lu qso in queue.\n", 
            process::this_pe, process::total_pes, local_queue.size());

        for (std::vector<OneQSOEstimate*>::iterator it = local_queue.begin(); it != local_queue.end(); ++it)
            (*it)->oneQSOiteration(powerspectra_fits, dbt_estimate_sum_before_fisher_vector, fisher_matrix_sum);

        thread_time = mytime::getTime() - thread_time;

        LOG::LOGGER.STD("Done for loop in %d/%d thread in %.2f minutes. Adding F and d critically.\n",
                process::this_pe, process::total_pes, thread_time);
        #endif

        // Calculation for each spectrum
        for (std::vector<OneQSOEstimate*>::iterator it = local_queue.begin(); it != local_queue.end(); ++it)
            (*it)->oneQSOiteration(powerspectra_fits, dbt_estimate_sum_before_fisher_vector, fisher_matrix_sum);

        // All reduce in MPI is enabled
        #if defined(ENABLE_MPI)
        MPI_Allreduce(MPI_IN_PLACE, fisher_matrix_sum, (bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS),
            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
            MPI_Allreduce(MPI_IN_PLACE, dbt_estimate_sum_before_fisher_vector[dbt_i], bins::TOTAL_KZ_BINS,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        #endif
        
        try
        {
            invertTotalFisherMatrix();
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR("ERROR while inverting Fisher matrix: %s.\n", e.what());
            
            delete [] powerspectra_fits;
            std::for_each(local_queue.begin(), local_queue.end(), std::default_delete<OneQSOEstimate>());

            throw e;
        }
        
        computePowerSpectrumEstimates();
        total_time_1it  = mytime::getTime() - total_time_1it;
        total_time     += total_time_1it;
        
        if (process::this_pe == 0)
            iterationOutput(fname_base, i, total_time_1it, total_time);

        if (hasConverged())
        {
            LOG::LOGGER.STD("Iteration has converged in %d iterations.\n", i+1);
            break;
        }

        try
        {
            _smoothPowerSpectra(powerspectra_fits);
        }
        catch (std::exception& e)
        {
            LOG::LOGGER.ERR("ERROR in Python script: %s\n", e.what());

            delete [] powerspectra_fits;
            std::for_each(local_queue.begin(), local_queue.end(), std::default_delete<OneQSOEstimate>());

            throw e;
        }
        
    }

    delete [] powerspectra_fits;
    std::for_each(local_queue.begin(), local_queue.end(), std::default_delete<OneQSOEstimate>());
}

bool OneDQuadraticPowerEstimate::hasConverged()
{
    double  diff, pMax, p1, p2, r, rfull, abs_mean = 0., abs_max = 0.;
    bool    bool_converged = true;
    int     kn, zm;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        p1 = fabs(current_power_estimate_vector[i_kz]);
        p2 = fabs(previous_power_estimate_vector[i_kz]);
        
        diff = fabs(p1 - p2);
        pMax = std::max(p1, p2);
        r    = diff / pMax;

        if (r > CONVERGENCE_EPS)    bool_converged = false;

        abs_mean += r / bins::DEGREE_OF_FREEDOM;
        abs_max   = std::max(r, abs_max);
    }
    
    LOG::LOGGER.STD("Mean relative change is %.1e.\n"
                    "Maximum relative change is %.1e. "
                    "Old test: Iteration converges when this is less than %.1e\n", 
                    abs_mean, abs_max, CONVERGENCE_EPS);
    
    // Perform a chi-square test as well
    // Maybe just do diagonal (dx)^2/F-1_ii
    
    mxhelp::vector_sub(previous_power_estimate_vector, current_power_estimate_vector, bins::TOTAL_KZ_BINS);

    r = 0;

    for (int i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)
        
        double  t = previous_power_estimate_vector[i_kz],
                e = inverse_fisher_matrix_sum[(1+bins::TOTAL_KZ_BINS) * i_kz];

        if (e < 0)  continue;

        r += (t*t) / e;
    }

    r  = sqrt(r / bins::DEGREE_OF_FREEDOM);

    rfull = sqrt(fabs(mxhelp::my_cblas_dsymvdot(previous_power_estimate_vector, fisher_matrix_sum, bins::TOTAL_KZ_BINS)) 
        / bins::DEGREE_OF_FREEDOM);
    
    LOG::LOGGER.TIME("%9.3e | %9.3e |\n", r, abs_mean);
    LOG::LOGGER.STD("Chi square convergence test: Diagonal Err: %.3f per dof. Full Fisher: %.3f per dof."
                    "Iteration converges when either is less than %.2f\n", 
                    r, rfull, specifics::CHISQ_CONVERGENCE_EPS);

    bool_converged = r < specifics::CHISQ_CONVERGENCE_EPS || rfull < specifics::CHISQ_CONVERGENCE_EPS;

    return bool_converged;
}

void OneDQuadraticPowerEstimate::writeFisherMatrix(const char *fname)
{
    gsl_matrix_view fisher_mv = gsl_matrix_view_array(fisher_matrix_sum, bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

    mxhelp::fprintfMatrix(fname, &fisher_mv.matrix);

    LOG::LOGGER.IO("Fisher matrix saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::writeSpectrumEstimates(const char *fname)
{
    FILE *toWrite;
    int i_kz, kn, zm;
    double z, k, p, e;

    toWrite = ioh::open_file(fname, "w");
    
    #ifdef LAST_K_EDGE
    fprintf(toWrite, "%d %d\n", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS-1);
    #else
    fprintf(toWrite, "%d %d\n", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);
    #endif

    auto fprint = [&](const int& zc) { fprintf(toWrite, "%d ", zc); };
    
    std::for_each(Z_BIN_COUNTS, Z_BIN_COUNTS+bins::NUMBER_OF_Z_BINS+2, fprint);

    // for (zm = 0; zm <= bins::NUMBER_OF_Z_BINS+1; ++zm)
    //     fprintf(toWrite, "%d ", Z_BIN_COUNTS[zm]);

    fprintf(toWrite, "\n");

    for (i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        // if (Z_BIN_COUNTS[zm+1] == 0)  continue;

        z = bins::ZBIN_CENTERS[zm];
        k = bins::KBAND_CENTERS[kn];
        p = current_power_estimate_vector[i_kz] + powerSpectrumFiducial(kn, zm);
        e = sqrt(inverse_fisher_matrix_sum[i_kz+bins::TOTAL_KZ_BINS*i_kz]);

        fprintf(toWrite, "%.3lf %e %e %e\n", z, k, p, e);
    }

    fclose(toWrite);
    
    LOG::LOGGER.IO("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::writeDetailedSpectrumEstimates(const char *fname)
{
    FILE *toWrite;
    int i_kz, kn, zm;
    double z, k1, k2, kc, Pfid, ThetaP, ErrorP, dk, bk, tk;

    toWrite = ioh::open_file(fname, "w");
    
    fprintf(toWrite, specifics::BUILD_SPECIFICS);
    specifics::printConfigSpecifics(toWrite);
    
    fprintf(toWrite, "# Fiducial Power Spectrum\n"
                     "# Pfid(k, z) = (A*pi/k0) * q^(2+n+alpha*ln(q)+beta*ln(x)) * x^B / (1 + lambda * k^2)\n"
                     "# k0=0.009 s km^-1, z0=3.0 and q=k/k0, x=(1+z)/(1+z0)\n"
                     "# Parameters set by config file:\n");
    fprintf(toWrite, "# A      = %15e\n"
                     "# n      = %15e\n"
                     "# alpha  = %15e\n"
                     "# B      = %15e\n"
                     "# beta   = %15e\n"
                     "# lambda = %15e\n", 
                     fidpd13::FIDUCIAL_PD13_PARAMS.A, fidpd13::FIDUCIAL_PD13_PARAMS.n, fidpd13::FIDUCIAL_PD13_PARAMS.alpha,
                     fidpd13::FIDUCIAL_PD13_PARAMS.B, fidpd13::FIDUCIAL_PD13_PARAMS.beta, fidpd13::FIDUCIAL_PD13_PARAMS.lambda);
    fprintf(toWrite, "# -----------------------------------------------------------------\n"
                     "# File Template\n"
                     "# Nz Nk\n"
                     "# n[0] n[1] ... n[Nz] n[Nz+1]\n"
                     "# z | k1 | k2 | kc | Pfid | ThetaP | Pest | ErrorP | d | b | t\n"
                     "# Nz                Number of redshift bins\n"
                     "# Nk                Number of k bins\n"
                     "# n[i]              Spectral chunk count in redshift bin i. Left-most and right-most are out of range\n"
                     "# z                 Redshift bin center\n"
                     "# k1                Lower edge of the k bin [s km^-1]\n"
                     "# k2                Upper edge of the k bin [s km^-1]\n"
                     "# kc                Center of the k bin [s km^-1]\n"
                     "# Pfid              Fiducial power at kc [km s^-1]\n"
                     "# ThetaP            Deviation from Pfid found by quadratic estimator = d - b - t [km s^-1]\n"
                     "# Pest              Pfid + ThetaP [km s^-1]\n"
                     "# ErrorP            Error estimated from diagonal terms of the inverse Fisher matrix [km s^-1]\n"
                     "# d                 Power estimate before noise (b) and fiducial power (t) subtracted [km s^-1]\n"
                     "# b                 Noise estimate [km s^-1]\n"
                     "# t                 Fiducial power estimate [km s^-1]\n"
                     "# -----------------------------------------------------------------\n");

    #ifdef LAST_K_EDGE
    fprintf(toWrite, "# %d %d\n# ", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS-1);
    #else
    fprintf(toWrite, "# %d %d\n# ", bins::NUMBER_OF_Z_BINS, bins::NUMBER_OF_K_BANDS);
    #endif

    for (zm = 0; zm <= bins::NUMBER_OF_Z_BINS+1; ++zm)
        fprintf(toWrite, "%d ", Z_BIN_COUNTS[zm]);

    fprintf(toWrite, "\n");
    fprintf(toWrite, "z %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n", 
        "k1", "k2", "kc", "Pfid", "ThetaP", "Pest", "ErrorP", "d", "b", "t");

    for (i_kz = 0; i_kz < bins::TOTAL_KZ_BINS; ++i_kz)
    {
        SKIP_LAST_K_BIN_WHEN_ENABLED(i_kz)

        bins::getFisherMatrixBinNoFromIndex(i_kz, kn, zm);   
        
        z  = bins::ZBIN_CENTERS[zm];
        
        k1 = bins::KBAND_EDGES[kn];
        k2 = bins::KBAND_EDGES[kn+1];
        kc = bins::KBAND_CENTERS[kn];

        Pfid = powerSpectrumFiducial(kn, zm);
        ThetaP = current_power_estimate_vector[i_kz];
        ErrorP = sqrt(inverse_fisher_matrix_sum[(1+bins::TOTAL_KZ_BINS)*i_kz]);

        dk = dbt_estimate_fisher_weighted_vector[0][i_kz];
        bk = dbt_estimate_fisher_weighted_vector[1][i_kz];
        tk = dbt_estimate_fisher_weighted_vector[2][i_kz];

        fprintf(toWrite, "%.3lf %15e %15e %15e %15e %15e %15e %15e %15e %15e %15e\n", 
                            z,  k1,  k2,  kc,  Pfid,ThetaP, Pfid+ThetaP, ErrorP, dk, bk, tk);
    }

    fclose(toWrite);
        
    LOG::LOGGER.IO("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
    LOG::LOGGER.STD("Quadratic 1D Power Spectrum estimate saved as %s.\n", fname);
}

void OneDQuadraticPowerEstimate::initializeIteration()
{
    for (int dbt_i = 0; dbt_i < 3; ++dbt_i)
        std::fill_n(dbt_estimate_sum_before_fisher_vector[dbt_i], bins::TOTAL_KZ_BINS, 0);

    std::fill_n(fisher_matrix_sum, bins::TOTAL_KZ_BINS*bins::TOTAL_KZ_BINS, 0);

    isFisherInverted = false;
}

void OneDQuadraticPowerEstimate::printfSpectra()
{
    int i_kz;

    for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS; ++zm)
        LOG::LOGGER.STD(" P(%.1f, k) |", bins::ZBIN_CENTERS[zm]);

    LOG::LOGGER.STD("\n");
    
    for (int kn = 0; kn < bins::NUMBER_OF_K_BANDS; ++kn)
    {
        for (int zm = 0; zm < bins::NUMBER_OF_Z_BINS; ++zm)
        {
            i_kz = bins::getFisherMatrixIndex(kn, zm);

            LOG::LOGGER.STD(" %10e |", current_power_estimate_vector[i_kz] + powerSpectrumFiducial(kn, zm));
        }
        
        LOG::LOGGER.STD("\n");
    }
}

double OneDQuadraticPowerEstimate::powerSpectrumFiducial(int kn, int zm)
{
    if (specifics::TURN_OFF_SFID)  return 0;
    
    return fidcosmo::fiducialPowerSpectrum(bins::KBAND_CENTERS[kn], bins::ZBIN_CENTERS[zm], &fidpd13::FIDUCIAL_PD13_PARAMS);
}

void OneDQuadraticPowerEstimate::iterationOutput(const char *fname_base, int it, double t1, double tot)
{
    char buf[500];
    printfSpectra();

    sprintf(buf, "%s_it%d_quadratic_power_estimate_detailed.dat", fname_base, it+1);
    writeDetailedSpectrumEstimates(buf);

    gsl_matrix_view fisher_mv    = gsl_matrix_view_array(fisher_matrix_sum, 
        bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);
    gsl_matrix_view invfisher_mv = gsl_matrix_view_array(inverse_fisher_matrix_sum, 
        bins::TOTAL_KZ_BINS, bins::TOTAL_KZ_BINS);

    sprintf(buf, "%s_it%d_fisher_matrix.dat", fname_base, it+1);
    mxhelp::fprintfMatrix(buf, &fisher_mv.matrix);
    sprintf(buf, "%s_it%d_inversefisher_matrix.dat", fname_base, it+1);
    mxhelp::fprintfMatrix(buf, &invfisher_mv.matrix);
    LOG::LOGGER.IO("Fisher matrix and inverse are saved as %s.\n", buf);

    LOG::LOGGER.STD("This iteration took %.1f minutes. Elapsed time so far is %.1f minutes.\n", 
        t1, tot);
    LOG::LOGGER.TIME("| %2d | %9.3e | %9.3e | ", it, t1, tot);

    mytime::printfTimeSpentDetails();
}




















