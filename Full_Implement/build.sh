#!/bin/bash
set -e

echo "========================================"
echo "Building Plane-Wave DFT (GPU Version)"
echo "========================================"

# ---------------------------------------------------------
# 1. Compile CUDA code
# ---------------------------------------------------------
echo "[1/3] Compiling CUDA..."
nvcc -c gpu_eigen.cu -o gpu_eigen.o -O3 -arch=sm_86
nvcc -c gpu_Vxc.cu -o gpu_Vxc.o -O3 -arch=sm_86

# ---------------------------------------------------------
# 2. Compile C++ code
# ---------------------------------------------------------
echo "[2/3] Compiling C++ (test_dft.cpp)..."
g++ test_dft.cpp -c -o test_dft.o -std=c++11 -O3 -fopenmp

# ---------------------------------------------------------
# 3. Link everything
# ---------------------------------------------------------
echo "[3/3] Linking...".]
nvcc test_dft.o gpu_eigen.o gpu_Vxc.o -o test_dft_exe \
    -Xcompiler -fopenmp \
    -L/usr/local/cuda/lib64 \
    -lcudart -lcusolver -lcublas \
    -lfftw3_omp -lfftw3 \
    -lopenblas \
    -lpthread

echo "========================================"
echo "Build completed!"
echo "========================================"

# ---------------------------------------------------------
# 4. Run
# ---------------------------------------------------------
echo ""
echo "Running program..."
echo ""
./test_dft_exe