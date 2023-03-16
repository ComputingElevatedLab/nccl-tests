/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "common.h"

void BruckGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks)
{
    *sendcount = (count / nranks) * nranks;
    *recvcount = (count / nranks) * nranks;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = count / nranks;
}

testResult_t BruckInitData(struct threadArgs *args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place)
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
    // We don't support in-place bruck
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void BruckGetBw(size_t count, int typesize, double sec, double *algBw, double *busBw, int nranks)
{
    double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)(nranks));
    *busBw = baseBw * factor;
}

int recursive_pow(int x, unsigned int p)
{
    if (p == 0)
    {
        return 1;
    }
    else if (p == 1)
    {
        return x;
    }

    int tmp = recursive_pow(x, p / 2);
    if (p % 2 == 0)
    {
        return tmp * tmp;
    }
    else
    {
        return x * tmp * tmp;
    }
}

std::vector<int> convert10tob(int w, int N, int b)
{
    std::vector<int> v(w);
    int i = 0;
    while (N)
    {
        v.at(i++) = (N % b);
        N /= b;
    }
    return v;
}

testResult_t BruckRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    int nRanks, rank;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    size_t rankOffset = count * wordSize(type);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
    printf("NCCL 2.7 or later is needed for bruck. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
    return testNcclError;
#else
    int radix = 2;
    int w = std::ceil(std::log(nRanks) / std::log(radix));
    int nlpow = recursive_pow(radix, w - 1);
    int d = (recursive_pow(radix, w) - nRanks) / nlpow;

    CUDACHECK(cudaMemcpy((char *)recvbuff, (char *)sendbuff, nRanks * rankOffset, cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaMemcpy(((char *)sendbuff) + ((nRanks - rank) * rankOffset), (char *)recvbuff, rank * rankOffset, cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaMemcpy((char *)sendbuff, ((char *)recvbuff) + (rank * rankOffset), (nRanks - rank) * rankOffset, cudaMemcpyDeviceToDevice));

    int sent_blocks[nlpow];
    int di = 0;
    int ci = 0;

    char *tempbuff;
    CUDACHECK(cudaMalloc((void **)&tempbuff, nlpow * rankOffset));

    int spoint = 1;
    int distance = 1;
    int next_distance = radix;
    for (int x = 0; x < w; x++)
    {
        for (int z = 1; z < radix; z++)
        {
            di = 0;
            ci = 0;
            spoint = z * distance;
            if (spoint > nRanks - 1)
            {
                break;
            }

            // get the sent data-blocks
            for (int i = spoint; i < nRanks; i += next_distance)
            {
                for (int j = i; j < (i + distance); j++)
                {
                    if (j > nRanks - 1)
                    {
                        break;
                    }
                    sent_blocks[di++] = j;
                    CUDACHECK(cudaMemcpy(((char *)tempbuff) + (rankOffset * ci++), ((char *)sendbuff) + (rankOffset * j), rankOffset, cudaMemcpyDeviceToDevice));
                }
            }

            int src = (rank + spoint) % nRanks;
            int dst = (rank - spoint + nRanks) % nRanks;

            NCCLCHECK(ncclGroupStart());
            NCCLCHECK(ncclSend((char *)tempbuff, di * count, type, dst, comm, stream));
            NCCLCHECK(ncclRecv((char *)recvbuff, di * count, type, src, comm, stream));
            NCCLCHECK(ncclGroupEnd());

            for (int i = 0; i < di; i++)
            {
                long long offset = sent_blocks[i] * rankOffset;
                CUDACHECK(cudaMemcpy(((char *)sendbuff) + offset, ((char *)recvbuff) + (i * rankOffset), rankOffset, cudaMemcpyDeviceToDevice));
            }
        }
        distance *= radix;
        next_distance *= radix;
    }

    CUDACHECK(cudaFree(tempbuff));

    for (int i = 0; i < nRanks; i++)
    {
        int index = (rank - i + nRanks) % nRanks;
        CUDACHECK(cudaMemcpy(((char *)recvbuff) + (index * rankOffset), ((char *)sendbuff) + (i * rankOffset), rankOffset, cudaMemcpyDeviceToDevice));
    }

    return testSuccess;
#endif
}

struct testColl bruckTest = {
    "Bruck",
    BruckGetCollByteCount,
    BruckInitData,
    BruckGetBw,
    BruckRunColl};

void BruckGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks)
{
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    BruckGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t BruckRunTest(struct threadArgs *args, int root, ncclDataType_t type, const char *typeName, ncclRedOp_t op, const char *opName)
{
    args->collTest = &bruckTest;
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

struct testEngine bruckEngine = {
    BruckGetBuffSize,
    BruckRunTest};

#pragma weak ncclTestEngine = bruckEngine
