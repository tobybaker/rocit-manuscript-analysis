#!/bin/bash
#SBATCH --job-name=trainsimple       
#SBATCH --partition=NC24adsA100 
#SBATCH --nodes=1
#SBATCH --time=3:30:00 
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --array=0-24

~/miniconda3/envs/pytorch_env/bin/python length_sample_training.py $SLURM_ARRAY_TASK_ID
