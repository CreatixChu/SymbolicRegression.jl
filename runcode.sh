#!/bin/bash
#SBATCH --job-name=JuliaSR                          # Job name
#SBATCH --time=0-18:00:00                           # Time limit hrs:min:sec
#SBATCH --ntasks=1                                  # Number of tasks (processes)
#SBATCH --cpus-per-task=20                          # Number of CPU cores per task (maximum on compute canada is 50)
#SBATCH --mem=8G                                    # Memory per node
#SBATCH --output=logs/slurm-%j-%N.out               # the print of xxx.jl will be logged in this file, %N for node name, %j for job id:w

module load StdEnv/2023 julia/1.11.3
julia --thread=64 muon_decay.jl
julia --thread=64 joint_pipeline.jl

# srun --cpus-per-task=1 --mem=16G --time=01:00:00 --pty bash 
# marginal 1000 nitr        take 4 min
# conditional 1000 niter    take 25 min