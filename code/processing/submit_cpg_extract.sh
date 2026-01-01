#!/bin/bash
#SBATCH --job-name=cpg_methylation
#SBATCH --partition=F32
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

# Usage: sbatch submit_cpg_extraction.sh <bam_file> <output_dir>
~/miniconda3/envs/pytorch_env/bin/python /hot/user/tobybaker/ROCIT_Paper/code/processing/extract_cpg_info_from_bam.py "$1" "$2" --workers 8