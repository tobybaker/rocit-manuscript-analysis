#!/bin/bash
#SBATCH --job-name=custominput       
#SBATCH --partition=NC24adsA100,NC24adsA100_8TB
#SBATCH --nodes=1
#SBATCH --time=12:30:00 
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --array=0-18

~/miniconda3/envs/pytorch_env/bin/python sample_training_custom_input.py $SLURM_ARRAY_TASK_ID
