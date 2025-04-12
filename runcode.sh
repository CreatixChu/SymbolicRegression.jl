#!/bin/bash
#SBATCH --job-name=CC-job              # Job name
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1              # Request 1 v100 GPUs
#SBATCH --mem=16G                      # Memory per node

module load StdEnv/2023 julia/1.11.3
JULIA_NUM_THREADS=4 julia muon_decay.jl