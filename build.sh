#!/bin/bash

clear
nvcc src/Kernel.cu src/Display.cpp src/RayTracer.cu &&
./a.out