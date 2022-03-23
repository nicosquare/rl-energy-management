#!/bin/bash
#
#SBATCH --job-name=test		# Job name
#SBATCH --output=output.%A_%a.txt	# Standard output and error log
#SBATCH --nodes=1                  	# Run all processes on a single node	
#SBATCH --ntasks=1			# Run on a single CPU
#SBATCH --mem-per-cpu=4000		# Job memory request
#SBATCH --gres=gpu:1			# Number of GPUs (per node)

srun python src/rl/td-a2c.py

