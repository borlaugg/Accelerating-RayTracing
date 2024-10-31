#!/bin/bash
for numThreads in $(seq 1 100)
do
  export OMP_NUM_THREADS=$numThreads
  { time build/inOneWeekend > image.ppm  ; } 2>> b.txt
done
