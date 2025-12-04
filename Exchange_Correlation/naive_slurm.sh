#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=gpu_vxc.output

cd $SLURM_SUBMIT_DIR
module load nvidia/cuda

nvcc -O3 -std=c++11 gpu_test_Vxc.cu -o gpu_test_Vxc

./gpu_test_Vxc