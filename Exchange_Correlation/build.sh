#!/bin/bash
set -e

echo "========================================"
echo "Building Vxc CPU vs GPU Comparison Test"
echo "========================================"

# ---------------------------------------------------------
# 1. Compile CUDA code
# ---------------------------------------------------------
echo "[1/3] Compiling CUDA (gpu_Vxc.cu)..."
nvcc -c gpu_Vxc.cu -o gpu_Vxc.o -O3 -arch=sm_86

# ---------------------------------------------------------
# 2. Compile C++ code
# ---------------------------------------------------------
echo "[2/3] Compiling C++ (test_Vxc.cpp)..."
g++ test_Vxc.cpp -c -o test_Vxc.o -std=c++11 -O3 -fopenmp

# ---------------------------------------------------------
# 3. Link everything
# ---------------------------------------------------------
echo "[3/3] Linking..."
nvcc test_Vxc.o gpu_Vxc.o -o test_vxc_exe \
    -Xcompiler -fopenmp \
    -L/usr/local/cuda/lib64 \
    -lcudart

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
./test_vxc_exe