// Tests mpisort::mergeSortedArrays. Run with any number of ranks, e.g.
//   mpirun -np 5 ./testMPIMergeSort
// Every rank can regenerate every other rank's data deterministically,
// so rank 0 compares the merged result against a serial sort.
#include <algorithm>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "core/mpi_manager.hpp"

#if defined(ENABLE_MPI)
#include "core/mpi_merge_sort.cpp"

using cpu_vec = std::vector<std::pair<double, int>>;

// Sorted local data of a PE. Sizes differ per PE; some PEs are empty.
// Duplicated keys test the tie-breaking on the int member.
cpu_vec makeLocalData(int pe, int scenario)
{
    std::mt19937 rng(1234 + 97 * pe + 7919 * scenario);
    int n;
    switch (scenario)
    {
    case 0: n = 10 + 3 * pe; break;          // varied sizes
    case 1: n = (pe % 2) ? 0 : 25; break;    // alternating empty PEs
    case 2: n = (pe == 0) ? 0 : 5; break;    // empty root
    default: n = 1; break;                   // single element each
    }

    std::uniform_int_distribution<int> key(0, 20);
    cpu_vec v(n);
    for (int i = 0; i < n; ++i)
        v[i] = std::make_pair(key(rng) * 0.5, pe * 1000 + i);
    std::sort(v.begin(), v.end());
    return v;
}

int main(int argc, char *argv[])
{
    mympi::init(argc, argv);
    int failures = 0;

    for (int scenario = 0; scenario < 4; ++scenario)
    {
        cpu_vec local = makeLocalData(mympi::this_pe, scenario);

        cpu_vec expected;
        for (int pe = 0; pe < mympi::total_pes; ++pe)
        {
            cpu_vec tmp = makeLocalData(pe, scenario);
            expected.insert(expected.end(), tmp.begin(), tmp.end());
        }
        std::sort(expected.begin(), expected.end());

        mpisort::mergeSortedArrays(0, mympi::total_pes, mympi::this_pe, local);

        if (mympi::this_pe == 0)
        {
            bool ok = (local == expected) && std::is_sorted(local.begin(), local.end());
            if (!ok)
            {
                ++failures;
                fprintf(stderr, "FAIL scenario %d: got %zu elements, expected %zu\n",
                    scenario, local.size(), expected.size());
            }
            else
                printf("PASS scenario %d (%zu elements, %d PEs)\n",
                    scenario, local.size(), mympi::total_pes);
        }

        // Also check bcast after merge gives everyone the same sorted vector
        mpisort::bcastCpuFnameVec(local);
        int bad = (local == expected) ? 0 : 1;
        mympi::allreduceInplace(&bad, 1);
        if (bad && mympi::this_pe == 0)
        {
            ++failures;
            fprintf(stderr, "FAIL scenario %d: broadcast result mismatch on %d PEs\n",
                scenario, bad);
        }
        mympi::barrier();
    }

    mympi::finalize();
    return failures;
}
#else
int main()
{
    printf("MPI disabled; skipping.\n");
    return 0;
}
#endif
