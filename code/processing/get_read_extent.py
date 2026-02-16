import sys
import pysam
import polars as pl
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from make_sample_training_data import get_bam_path

from rocit.constants import HUMAN_CHROMOSOMES,HUMAN_CHROMOSOME_ENUM
from pathlib import Path
              
def process_chromosome( bam_path, chromosome):
    """Process a single chromosome and return read data"""
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    read_store = []
    
    try:
        # Fetch reads only for this specific chromosome
        for read in bamfile.fetch(chromosome):
            # Skip secondary alignments
            if read.is_secondary or read.is_supplementary or read.is_unmapped:
                continue
            
            positions = read.get_reference_positions()
            if not positions:  # Skip if no positions available
                continue
            
            read_entry = {'read_index':read.query_name,'chromosome':read.reference_name,'reference_start':min(positions),'reference_end':max(positions),'read_length':len(read.query_sequence)}
            read_store.append(read_entry)
        
    except Exception as e:
        print(f"Error processing chromosome {chromosome}: {e}")
        return pl.DataFrame()
    finally:
        bamfile.close()
    
    df =  pl.DataFrame(read_store)
    df = df.with_columns(pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM),pl.col('reference_start').cast(pl.Int32),pl.col('reference_end').cast(pl.Int32),pl.col('read_length').cast(pl.Int32))
    return df
def main():
    sample_ids = ['BS14772_TU','BS15145_TU','053_TU','192_TU','216_TU','244_TU','264_TU']
    sample_ids = sample_ids+ ['BS14772_NL','BS15145_NL','053_NL','192_NL','216_NL','244_NL','264_NL']
    sample_id = sample_ids[int(sys.argv[1])]
    sample_bam_path = get_bam_path(sample_id)

    # Use all available CPU cores (or specify a number)
    max_workers = min(len(HUMAN_CHROMOSOMES), mp.cpu_count()-1)
    
    
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers,mp_context=ctx) as executor:
        dataframes = executor.map(process_chromosome,[sample_bam_path] * len(HUMAN_CHROMOSOMES),HUMAN_CHROMOSOMES)
    

    print(dataframes)
    combined_data = pl.concat(dataframes)
    print(combined_data)
    
    
    
    # Save combined results
    out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/output/read_extent')
    output_path = out_dir/f'{sample_id}_read_extent.parquet'
    combined_data.write_parquet(output_path)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
