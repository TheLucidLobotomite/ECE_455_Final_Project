#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=eigen_all.output

cd $SLURM_SUBMIT_DIR
module load nvidia/cuda

echo "Compiling naive_gpu_eigen.cu..."
nvcc naive_gpu_eigen.cu -o naive_gpu_eigen

echo "Running naive_gpu_eigen..."
./naive_gpu_eigen

echo "Compiling test_naive_gpu_eigen.cu..."
nvcc test_naive_gpu_eigen.cu -o test_naive_gpu_eigen

echo "Running test_naive_gpu_eigen..."
./test_naive_gpu_eigen

echo "All tasks completed."
