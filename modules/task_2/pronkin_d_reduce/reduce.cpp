// Copyright 2020 Pronkin Dmitry
#include <mpi.h>
#include <memory>
#include <iostream>
#include "../../../modules/task_2/pronkin_d_reduce/reduce.h"

int Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm) {
	if (sendbuf == nullptr || recvbuf == nullptr || sendbuf == recvbuf)
		return MPI_ERR_BUFFER;
	if (count <= 0 )
		return MPI_ERR_COUNT;
	if (op != MPI_MAX && op != MPI_MIN && op != MPI_SUM && op != MPI_PROD)
		return MPI_ERR_OP;
	if (comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF)
		return MPI_ERR_COMM;

	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	void* buffer;

	if (type == MPI_INT) {
		buffer = new int[count];
		rank == root ? std::memcpy(buffer, sendbuf, count * sizeof(int)) : 0;
		std::memcpy(recvbuf, sendbuf, count * sizeof(int));
	} else if (type == MPI_FLOAT) {
		buffer = new float[count];
		rank == root ? std::memcpy(buffer, sendbuf, count * sizeof(float)) : 0;
		std::memcpy(recvbuf, sendbuf, count * sizeof(float));
	} else if (type == MPI_DOUBLE) {
		buffer = new double[count];
		rank == root ? std::memcpy(buffer, sendbuf, count * sizeof(double)) : 0;
		std::memcpy(recvbuf, sendbuf, count * sizeof(double));
	} else
		return MPI_ERR_TYPE;

	if (rank == root) {
		for (int proc = 0; proc < size; proc++) {
			MPI_Status status;
			if (proc != root)
				MPI_Recv(buffer, count, type, proc, 0, comm, &status);	
			if (type == MPI_INT) {
				if (op == MPI_MAX) {
					for (int i = 0; i < count; i++)
						if (static_cast<int*>(recvbuf)[i] < static_cast<int*>(buffer)[i])
							static_cast<int*>(recvbuf)[i] = static_cast<int*>(buffer)[i];
				} else if (op == MPI_MIN) {
					for (int i = 0; i < count; i++)
						if (static_cast<int*>(recvbuf)[i] > static_cast<int*>(buffer)[i])
							static_cast<int*>(recvbuf)[i] = static_cast<int*>(buffer)[i];
				} else if (op == MPI_SUM) {
					for (int i = 0; i < count; i++) 
						static_cast<int*>(recvbuf)[i] += static_cast<int*>(buffer)[i];
				} else if (op == MPI_PROD) {
					for (int i = 0; i < count; i++)
						static_cast<int*>(recvbuf)[i] *= static_cast<int*>(buffer)[i];
				}
			}
			if (type == MPI_FLOAT) {
				if (op == MPI_MAX) {
					for (int i = 0; i < count; i++)
						if (static_cast<float*>(recvbuf)[i] < static_cast<float*>(buffer)[i])
							static_cast<float*>(recvbuf)[i] = static_cast<float*>(buffer)[i];
				} else if (op == MPI_MIN) {
					for (int i = 0; i < count; i++)
						if (static_cast<float*>(recvbuf)[i] > static_cast<float*>(buffer)[i])
							static_cast<float*>(recvbuf)[i] = static_cast<float*>(buffer)[i];
				} else if (op == MPI_SUM) {
					for (int i = 0; i < count; i++)
						static_cast<float*>(recvbuf)[i] += static_cast<float*>(buffer)[i];
				} else if (op == MPI_PROD) {
					for (int i = 0; i < count; i++)
						static_cast<float*>(recvbuf)[i] *= static_cast<float*>(buffer)[i];
				}
			}
			if (type == MPI_DOUBLE) {
				if (op == MPI_MAX) {
					for (int i = 0; i < count; i++)
						if (static_cast<double*>(recvbuf)[i] < static_cast<double*>(buffer)[i])
							static_cast<double*>(recvbuf)[i] = static_cast<double*>(buffer)[i];
				} else if (op == MPI_MIN) {
					for (int i = 0; i < count; i++)
						if (static_cast<double*>(recvbuf)[i] > static_cast<double*>(buffer)[i])
							static_cast<double*>(recvbuf)[i] = static_cast<double*>(buffer)[i];
				} else if (op == MPI_SUM) {
					for (int i = 0; i < count; i++)
						static_cast<double*>(recvbuf)[i] += static_cast<double*>(buffer)[i];
				} else if (op == MPI_PROD) {
					for (int i = 0; i < count; i++)
						static_cast<double*>(recvbuf)[i] *= static_cast<double*>(buffer)[i];
				}
			}
		}
	} else {
		MPI_Send(sendbuf, count, type, root, 0, comm);
	}

	if (type == MPI_INT)
        delete[] static_cast<int*>(buffer);
    else if (type == MPI_FLOAT)
        delete[] static_cast<float*>(buffer);
    else if (type == MPI_DOUBLE)
        delete[] static_cast<double*>(buffer);

    return MPI_SUCCESS;
}