/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

void SpreadoutGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks)
{
    *sendcount = (count / nranks) * nranks;
    *recvcount = (count / nranks) * nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count / nranks;
}

testResult_t SpreadoutInitData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place)
{
    size_t sendcount = args->sendBytes / wordSize(type);
    size_t recvcount = args->expectedBytes / wordSize(type);
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++)
    {
        CUDACHECK(cudaSetDevice(args->gpus[i]));
        int rank = ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));
        void *data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
        TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33 * rep + rank, 1, 0));
        for (int j = 0; j < nranks; j++)
        {
            size_t partcount = sendcount / nranks;
            TESTCHECK(InitData((char *)args->expected[i] + j * partcount * wordSize(type), partcount, rank * partcount, type, ncclSum, 33 * rep + j, 1, 0));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }
    // We don't support in-place spreadout
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void SpreadoutGetBw(size_t count, int typesize, double sec, double *algBw, double *busBw, int nranks)
{
    double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)(nranks));
    *busBw = baseBw * factor;
}

testResult_t SpreadoutRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    int nRanks, rank;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    size_t rankOffset = count * wordSize(type);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf("NCCL 2.7 or later is needed for spreadout. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#else
    ncclGroupStart();
    for (int r = 0; r < nRanks; r++)
    {
        int src = (rank + r) % nRanks;
        ncclRecv(((char *)recvbuff) + (src * rankOffset), count, type, src, comm, stream);
    }
    for (int r = 0; r < nRanks; r++)
    {
        int dst = (rank - r + nRanks) % nRanks;
        ncclSend(((char *)sendbuff) + (dst * rankOffset), count, type, dst, comm, stream);
    }
    ncclGroupEnd();
    return testSuccess;
#endif
}

struct testColl spreadoutTest = {
    "Spreadout",
    SpreadoutGetCollByteCount,
    SpreadoutInitData,
    SpreadoutGetBw,
    SpreadoutRunColl};

void SpreadoutGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    SpreadoutGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t SpreadoutRunTest(struct threadArgs *args, int root, ncclDataType_t type, const char *typeName, ncclRedOp_t op, const char *opName)
{
    args->collTest = &spreadoutTest;
    ncclDataType_t *run_types;
    const char **run_typenames;
    int type_count;

    if ((int)type != -1)
    {
        type_count = 1;
        run_types = &type;
        run_typenames = &typeName;
    }
    else
    {
        type_count = test_typenum;
        run_types = test_types;
        run_typenames = test_typenames;
    }

    for (int i = 0; i < type_count; i++)
    {
        TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
    }
    return testSuccess;
}

struct testEngine spreadoutEngine = {
    SpreadoutGetBuffSize,
    SpreadoutRunTest};

#pragma weak ncclTestEngine = spreadoutEngine
