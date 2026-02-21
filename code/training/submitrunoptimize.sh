#!/bin/bash
#SBATCH --job-name=runsim        # Job name
#SBATCH --partition=NC24adsA100,NC24adsA100_8TB # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --time=12:00:00            # Maximum runtime (D-HH:MM:SS)
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --array=0-5

SAMPLE_IDS=('BS14772_TU' 'BS15145_TU' '216_TU' '244_TU' '264_TU' '053_TU')
CURRENT_SAMPLE=${SAMPLE_IDS[$SLURM_ARRAY_TASK_ID]}

~/miniconda3/envs/pytorch_env/bin/python optimize_reads.py $CURRENT_SAMPLE