#!/bin/bash

echo " >>> Compiling... "
nvcc src/Kernel.cu src/Display.cpp src/RayTracer.cu -lcurand &&

echo " >>> Running program... " &&
./a.out

# clear
# rm image.ppm
# nvcc src/Kernel.cu src/Display.cpp src/RayTracer.cu &&
# clear &&
# ./a.out
