#!/bin/bash
/usr/local/cuda-12.6/bin/nvcc -ccbin g++  -m64 -gencode arch=compute_86,code=sm_86 -o out.o -c main.cu
/usr/local/cuda-12.6/bin/nvcc -ccbin g++  -m64 -gencode arch=compute_86,code=sm_86 -o out out.o
