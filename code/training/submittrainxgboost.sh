#!/bin/bash
#SBATCH --job-name=xgtrain
#SBATCH --partition=F32 # 
#SBATCH --nodes=1    
#SBATCH --time=24:00:00           
#SBATCH --mem=0
#SBATCH --array=0-5%3
#SBATCH --exclusive

~/miniconda3/envs/pytorch_env/bin/python train_xgboost.py $SLURM_ARRAY_TASK_ID