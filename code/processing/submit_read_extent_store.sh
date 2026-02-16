#!/bin/bash
#SBATCH --job-name=read_variant
#SBATCH --partition=F16
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --array=0-12

~/miniconda3/envs/pytorch_env/bin/python get_read_extent.py $SLURM_ARRAY_TASK_ID
