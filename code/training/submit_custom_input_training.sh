#!/bin/bash
#SBATCH --job-name=custominput       
#SBATCH --partition=NC24adsA100 
#SBATCH --nodes=1
#SBATCH --time=12:30:00 
#SBATCH --mem=70G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-12

~/miniconda3/envs/pytorch_env/bin/python sample_training_custom_input.py $SLURM_ARRAY_TASK_ID
