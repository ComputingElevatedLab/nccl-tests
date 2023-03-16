#!/bin/bash
make CC=gcc-9 CXX=g++-9 LD=g++-9 MPI_HOME=/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0 CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda
mpirun -np 2 ./build/alltoall_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1
mpirun -np 2 ./build/bruck_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1
mpirun -np 2 ./build/spreadout_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1