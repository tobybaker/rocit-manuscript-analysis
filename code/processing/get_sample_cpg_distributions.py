import polars as pl
import sys
from pathlib import Path
from make_sample_training_data import get_bam_path
from rocit.preprocessing import process_cpg_distribution
if __name__ =='__main__':
    sample_ids = ['BS14772_TU','BS15145_TU','053_TU','216_TU','244_TU','264_TU']

    sample_id = sample_ids[int(sys.argv[1])]


    output_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/cpg_methylation_distribution')
    
    in_dir = Path(f'/hot/user/tobybaker/ROCIT_Paper/input_data/cpg_methylation/{sample_id}')
    process_cpg_distribution.get_aggregate_methylation_distribution_from_dir(in_dir,output_dir,sample_id)

