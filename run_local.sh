#!/bin/bash
make CC=gcc-9 CXX=g++-9 LD=g++-9 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/lib/cuda-11.2
mpirun -np 1 ./build/alltoall_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1
mpirun -np 1 ./build/bruck_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1
mpirun -np 1 ./build/spreadout_perf -a 3 -c 1 -d all -b 8 -e 128M -f 2 -g 1