#!/bin/bash
set -e

echo "========================================"
echo "Building Eigensolver CPU vs GPU Test"
echo "========================================"

# ---------------------------------------------------------
# 1. Compile CUDA code
# ---------------------------------------------------------
echo "[1/3] Compiling CUDA (gpu_eigen.cu)..."
nvcc -c gpu_eigen.cu -o gpu_eigen.o -O3 -arch=sm_86

# ---------------------------------------------------------
# 2. Compile C++ code
# ---------------------------------------------------------
echo "[2/3] Compiling C++ (test_eigen.cpp)..."
g++ test_eigen.cpp -c -o test_eigen.o -std=c++11 -O3 -fopenmp

# ---------------------------------------------------------
# 3. Link everything
# ---------------------------------------------------------
echo "[3/3] Linking..."
nvcc test_eigen.o gpu_eigen.o -o test_eigen_exe \
    -Xcompiler -fopenmp \
    -L/usr/local/cuda/lib64 \
    -lcudart -lcusolver -lopenblas

echo "========================================"
echo "Build completed!"
echo "========================================"

# ---------------------------------------------------------
# 4. Run
# ---------------------------------------------------------
echo ""
echo "Running comparison tests..."
echo ""
export OMP_NUM_THREADS=8
./test_eigen_exe