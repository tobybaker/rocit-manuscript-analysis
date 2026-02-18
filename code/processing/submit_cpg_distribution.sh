#!/bin/bash
#SBATCH --job-name=sample_dist
#SBATCH --partition=F72
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --array=0-5

~/miniconda3/envs/pytorch_env/bin/python get_sample_cpg_distributions.py $SLURM_ARRAY_TASK_ID