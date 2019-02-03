#!/bin/bash
#
#SBATCH --job-name HYPOTHESIS-UNIVARIATE
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=joeri.hermans@doct.uliege.be
#SBATCH --comment=P-AVO
#SBATCH --time=300

# Parallelism settings.
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export KMP_AFFINITY=scatter

# Run the optimization procedure.
srun --export=all python train.py --output="models_$1" --lower="-10" --upper="10"
