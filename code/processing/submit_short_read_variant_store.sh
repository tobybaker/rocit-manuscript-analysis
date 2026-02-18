#!/bin/bash
#SBATCH --job-name=short_var
#SBATCH --partition=M64
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --array=0-5

~/miniconda3/envs/pytorch_env/bin/python get_read_variant_store.py $SLURM_ARRAY_TASK_ID short_read
