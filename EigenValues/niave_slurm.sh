#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=eigen_all.output

cd $SLURM_SUBMIT_DIR
module load nvidia/cuda

echo "Compiling both CUDA sources into one binary..."
nvcc naive_gpu_eigen.cu test_naive_gpu_eigen.cu -o gpu_eigen \
    -lcusolver -lcublas

echo "Running gpu_eigen..."
./gpu_eigen

echo "All tasks completed."
