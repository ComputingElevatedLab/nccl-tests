#!/bin/bash
#COBALT -n 10 -t 0:30:00 -q full-node -A dist_relational_alg --attrs filesystems=home,grand,theta-fs0

NCCL_DEBUG=INFO
module load nccl
make CC=gcc-9 CXX=g++-9 LD=g++-9 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ CUDA_HOME=/usr/local/cuda-11.4 NCCL_HOME=/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.12.12-1_gcc-9.4.0-1ubuntu1-20.04

NODES=$(echo $COBALT_NODEFILE | wc -l)
NPROC=$(($NODES * 8))

mpirun -hostfile $COBALT_NODEFILE -n $NPROC -npernode 8 ./build/alltoall_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1
mpirun -hostfile $COBALT_NODEFILE -n $NPROC -npernode 8 ./build/bruck_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1
mpirun -hostfile $COBALT_NODEFILE -n $NPROC -npernode 8 ./build/spreadout_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1