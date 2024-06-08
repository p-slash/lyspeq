#define MPI_VEC_TAG 2
#define MPI_SIZE_TAG 33

namespace mpisort
{
    // Ref: https://stackoverflow.com/questions/33618937/trouble-understanding-mpi-type-create-struct
    MPI_Datatype getPairType()
    {
        // Create MPI pair type
        std::pair<double, int> tmp = std::make_pair(10.1, 101);

        int blocklengths[] = {1, 1};
        MPI_Aint disp[3];
        MPI_Get_address(&tmp, &disp[0]);
        MPI_Get_address(&tmp.first, &disp[1]);
        MPI_Get_address(&tmp.second, &disp[2]);

        MPI_Aint offsets[2] = { MPI_Aint_diff(disp[1], disp[0]), MPI_Aint_diff(disp[2], disp[0]) };

        MPI_Aint lb, extent;
        MPI_Datatype types[] = {MPI_DOUBLE, MPI_INT};
        MPI_Datatype tmp_type, my_mpi_pair_type;

        MPI_Type_create_struct(2, blocklengths, offsets, types, &tmp_type);
        MPI_Type_get_extent(tmp_type, &lb, &extent);
        MPI_Type_create_resized(tmp_type, lb, extent, &my_mpi_pair_type);
        MPI_Type_commit(&my_mpi_pair_type);
        MPI_Type_free(&tmp_type);

        return my_mpi_pair_type;
    }

    void bcastCpuFnameVec(std::vector<std::pair<double, int>> &cpu_index_vec, int root_pe=0)
    {
        MPI_Datatype MY_MPI_PAIR = getPairType();
        int size = cpu_index_vec.size();

        // Root sends the size of the vector
        MPI_Bcast(&size, 1, MPI_INT, root_pe, MPI_COMM_WORLD);
        // Everybody resizes accordingly
        // Root doesn't do anything as it already has the right size
        cpu_index_vec.resize(size);

        MPI_Bcast(cpu_index_vec.data(), size, MY_MPI_PAIR, root_pe, MPI_COMM_WORLD);
        MPI_Type_free(&MY_MPI_PAIR);
    }

    void mergeSortedArrays(int height, int Npe, int id, 
        std::vector<std::pair<double, int>> &local_cpu_ind_vec)
    {
        if (Npe == 1)  // We have reached the end
            return;

        int parent_pe, child_pe, next_Npe, local_size = local_cpu_ind_vec.size();
        int transmission_count = local_size;
        MPI_Status status;
        MPI_Datatype MY_MPI_PAIR = getPairType();

        next_Npe = (Npe + 1) / 2;

        // Given a height, parent PEs are 2**(height+1), 1 << (height+1), 
        // height starts from 0 at the bottom (all PEs)
        // This means e.g. at height 0 we need to map: 3->2, 2->2 and 5->4, 4->4 to find parents
        parent_pe = (id & ~(1 << height));

        if (id == parent_pe)
        {
            // If this is the parent PE, receive from the child.
            child_pe = (id | (1 << height));

            // If childless, carry on to the next cycle
            if (child_pe >= mympi::total_pes)
            {
                mergeSortedArrays(height+1, next_Npe, id, local_cpu_ind_vec);
                return;
            }

            // First recieve the size of the transmission
            MPI_Recv(&transmission_count, 1, MPI_INT, child_pe, MPI_SIZE_TAG,
                MPI_COMM_WORLD, &status);
            // MPI_Probe(child_pe, MPI_VEC_TAG, MPI_COMM_WORLD, &status);
            // MPI_Get_count(&status, MY_MPI_PAIR, &transmission_count);

            local_cpu_ind_vec.resize(local_size+transmission_count);

            MPI_Recv(local_cpu_ind_vec.data()+local_size, transmission_count, MY_MPI_PAIR, child_pe, 
                MPI_VEC_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            std::inplace_merge(local_cpu_ind_vec.begin(), local_cpu_ind_vec.begin()+local_size, 
                local_cpu_ind_vec.end());

            // Recursive call 
            mergeSortedArrays(height+1, next_Npe, id, local_cpu_ind_vec);
        }
        else
        {
            MPI_Send(&transmission_count, 1, MPI_INT, parent_pe, MPI_SIZE_TAG,
                MPI_COMM_WORLD);
            // If this is the child, just send data to the parent
            MPI_Send(local_cpu_ind_vec.data(), transmission_count, MY_MPI_PAIR, parent_pe, 
                MPI_VEC_TAG, MPI_COMM_WORLD);
        }

        MPI_Type_free(&MY_MPI_PAIR);
    }
}

#undef MPI_SIZE_TAG
#undef MPI_VEC_TAG
