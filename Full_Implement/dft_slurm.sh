#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=dft_gpu.output

cd $SLURM_SUBMIT_DIR
module load nvidia/cuda

nvcc test_dft_gpu.cu -o dft_gpu \
    -lcusolver -lcublas -lfftw3 -lfftw3_threads -lopenblas \
    -Xcompiler -fopenmp

./dft_gpu