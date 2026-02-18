import sys
import polars as pl

from pathlib import Path
from make_sample_training_data import get_bam_path
from rocit.preprocessing import extract_pacbio_cpg_info
if __name__ =='__main__':
    sample_ids = ['BS14772_TU','BS15145_TU','053_TU','192_TU','216_TU','244_TU','264_TU']
    sample_ids = sample_ids+ ['BS14772_NL','BS15145_NL','053_NL','192_NL','216_NL','244_NL','264_NL']
    sample_id = sample_ids[int(sys.argv[1])]
    sample_bam_path = get_bam_path(sample_id)

    base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/cpg_methylation')
    output_dir = base_dir/sample_id
    output_dir.mkdir(exist_ok=True,parents=True)

    extract_pacbio_cpg_info.process_bam(sample_bam_path,output_dir,sample_id)

