#!/bin/bash
#SBATCH --job-name=trainsimple       
#SBATCH --partition=NC24adsA100 
#SBATCH --nodes=1
#SBATCH --time=12:30:00 
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --array=0-12
#SBATCH --exclusive

~/miniconda3/envs/pytorch_env/bin/python main_sample_training.py $SLURM_ARRAY_TASK_ID
