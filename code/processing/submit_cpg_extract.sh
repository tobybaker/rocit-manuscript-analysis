#!/bin/bash
#SBATCH --job-name=cpg_methylation
#SBATCH --partition=M64
#SBATCH --cpus-per-task=8
#SBATCH --mem=150GB
#SBATCH --time=24:00:00
#SBATCH --array=12-13
~/miniconda3/envs/pytorch_env/bin/python extract_tumor_cpg_data.py $SLURM_ARRAY_TASK_ID
