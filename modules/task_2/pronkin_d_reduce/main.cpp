// Copyright 2020 Pronkin Dmitry
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <ctime>
#include "./reduce.h"

#define EPSILON 0.0001

TEST(Parallel_Operations_MPI, MPI_Err_Buffer) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* sendbuf = nullptr;
    int* recvbuf = nullptr;
    const int count = 1;

    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD), MPI_ERR_BUFFER);
}

TEST(Parallel_Operations_MPI, MPI_Err_Count) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* sendbuf = new int(13);
    int* recvbuf = new int(13);
    const int count = -1;

    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD), MPI_ERR_COUNT);
}

TEST(Parallel_Operations_MPI, MPI_Err_Op) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* sendbuf = new int(13);
    int* recvbuf = new int(13);
    const int count = 1;

    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD), MPI_ERR_OP);
}

TEST(Parallel_Operations_MPI, MPI_Err_Comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* sendbuf = new int(13);
    int* recvbuf = new int(13);
    const int count = 1;
    MPI_Comm comm = MPI_COMM_NULL;

    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_MAX, 0, comm), MPI_ERR_COMM);
}

TEST(Parallel_Operations_MPI, MPI_Err_Type) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* sendbuf = new int(13);
    int* recvbuf = new int(13);
    const int count = 1;

    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_BYTE, MPI_MAX, 0, MPI_COMM_WORLD), MPI_ERR_TYPE);
}

TEST(Parallel_Operations_MPI, IntMax) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf = rank + 1;
    int recvbuf = 0;
    const int count = 1;
    int root = 0;
    const int max = size;

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_INT, MPI_MAX, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_EQ(recvbuf, max);
    }
}

TEST(Parallel_Operations_MPI, IntMin) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf = rank + 1;
    int recvbuf = 0;
    const int count = 1;
    const int root = 0;
    const int min = 1;

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_EQ(recvbuf, min);
    }
}

TEST(Parallel_Operations_MPI, IntSum) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf = rank + 1;
    int recvbuf = 0;
    const int count = 1;
    const int root = 0;
    const int sum = ((1 + size) * size) / 2;

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_EQ(recvbuf, sum);
    }
}

TEST(Parallel_Operations_MPI, IntProd) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf = rank + 1;
    int recvbuf = 0;
    const int count = 1;
    const int root = 0;
    int prod = 1;
    for (int i = 1; i <= size; i++)
        prod *= i;

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_INT, MPI_PROD, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_EQ(recvbuf, prod);
    }
}

TEST(Parallel_Operations_MPI, FloatMax) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float sendbuf = static_cast<float>(rank + 1);
    float recvbuf = 0.0f;
    const int count = 1;
    const int root = 0;
    const float max = static_cast<float>(size);

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_FLOAT, MPI_MAX, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, max, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, FloatMin) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float sendbuf = static_cast<float>(rank + 1);
    float recvbuf = 0.0f;
    const int count = 1;
    const int root = 0;
    const float min = static_cast<float>(1);

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_FLOAT, MPI_MIN, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, min, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, FloatSum) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float sendbuf = static_cast<float>(rank + 1);
    float recvbuf = 0.0f;
    const int count = 1;
    const int root = 0;
    const float sum = static_cast<float>(((1 + size) * size) / 2);

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, sum, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, FloatProd) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float sendbuf = static_cast<float>(rank + 1) * 0.01f;
    float recvbuf = 0.0f;
    const int count = 1;
    const int root = 0;
    float prod = 1.0f;
    for (int i = 1; i <= size; i++)
        prod *= static_cast<float>(i) * 0.01f;

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_FLOAT, MPI_PROD, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, prod, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, DoubleMax) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double sendbuf = static_cast<double>(rank + 1);
    double recvbuf = 0.0;
    const int count = 1;
    const int root = 0;
    const double max = static_cast<double>(size);

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, max, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, DoubleMin) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double sendbuf = static_cast<double>(rank + 1);
    double recvbuf = 0.0;
    const int count = 1;
    const int root = 0;
    const double min = static_cast<double>(1);

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, min, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, DoubleSum) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double sendbuf = static_cast<double>(rank + 1);
    double recvbuf = 0.0;
    const int count = 1;
    const int root = 0;
    const double sum = static_cast<double>(((1 + size) * size) / 2);

    ASSERT_EQ(Reduce(&sendbuf, &recvbuf, count, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, sum, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, DoubleProd) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* sendbuf = new double(static_cast<double>(rank + 1) * 0.01);
    double recvbuf = 0.0;
    const int count = 1;
    const int root = 0;
    double prod = 1.0;
    for (int i = 1; i <= size; i++)
        prod *= static_cast<double>(i) * 0.01;

    ASSERT_EQ(Reduce(sendbuf, &recvbuf, count, MPI_DOUBLE, MPI_PROD, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        ASSERT_NEAR(recvbuf, prod, EPSILON);
    }
}

TEST(Parallel_Operations_MPI, IntArrMax) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int count = 7;
    int* sendbuf = new int[count];
    int* recvbuf = new int[count];
    int* example = new int[count];
    const int root = 0;

    for (int i = 0; i < count; i++)
        sendbuf[i] = gen() % 10;

    ASSERT_EQ(MPI_Reduce(sendbuf, example, count, MPI_INT, MPI_MAX, root, MPI_COMM_WORLD), MPI_SUCCESS);
    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_MAX, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        for (int i = 0; i < count; i++)
            ASSERT_EQ(recvbuf[i], example[i]);
    }
}

TEST(Parallel_Operations_MPI, IntArrMin) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int count = 7;
    int* sendbuf = new int[count];
    int* recvbuf = new int[count];
    int* example = new int[count];
    const int root = 0;

    for (int i = 0; i < count; i++)
        sendbuf[i] = gen() % 10;

    ASSERT_EQ(MPI_Reduce(sendbuf, example, count, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD), MPI_SUCCESS);
    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        for (int i = 0; i < count; i++)
            ASSERT_EQ(recvbuf[i], example[i]);
    }
}

TEST(Parallel_Operations_MPI, FloatArrSum) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int count = 7;
    float* sendbuf = new float[count];
    float* recvbuf = new float[count];
    float* example = new float[count];
    const int root = 0;

    for (int i = 0; i < count; i++)
        sendbuf[i] = (gen() % 10) * 0.1f;

    ASSERT_EQ(MPI_Reduce(sendbuf, example, count, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD), MPI_SUCCESS);
    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        for (int i = 0; i < count; i++)
            ASSERT_NEAR(recvbuf[i], example[i], EPSILON);
    }
}

TEST(Parallel_Operations_MPI, DoubleArrProd) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int count = 7;
    double* sendbuf = new double[count];
    double* recvbuf = new double[count];
    double* example = new double[count];
    const int root = 0;

    for (int i = 0; i < count; i++)
        sendbuf[i] = (gen() % 10) * 0.1;

    ASSERT_EQ(MPI_Reduce(sendbuf, example, count, MPI_DOUBLE, MPI_PROD, root, MPI_COMM_WORLD), MPI_SUCCESS);
    ASSERT_EQ(Reduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_PROD, root, MPI_COMM_WORLD), MPI_SUCCESS);
    if (rank == root) {
        for (int i = 0; i < count; i++)
            ASSERT_NEAR(recvbuf[i], example[i], EPSILON);
    }
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
