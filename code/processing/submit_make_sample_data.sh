#!/bin/bash
#SBATCH --job-name=sampledata
#SBATCH --partition=F72
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --exclusive


~/miniconda3/envs/pytorch_env/bin/python make_sample_training_data.py $SLURM_ARRAY_TASK_ID
