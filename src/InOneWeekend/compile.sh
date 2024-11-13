#!/bin/bash
/usr/local/cuda/bin/nvcc -ccbin g++  -m64 -gencode arch=compute_75,code=sm_75 -o out.o -c main.cu
/usr/local/cuda/bin/nvcc -ccbin g++  -m64 -gencode arch=compute_75,code=sm_75 -o out out.o