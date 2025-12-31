#!/bin/bash
#SBATCH --job-name=cpg_methylation
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

# Usage: sbatch submit_cpg_extraction.sh <bam_file> <output_dir>
python extract_cpg_methylation.py "$1" "$2" --workers 12