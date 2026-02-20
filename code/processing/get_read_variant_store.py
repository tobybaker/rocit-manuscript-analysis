import sys
import polars as pl
from pathlib import Path
import multiprocessing as mp
from rocit.preprocessing import bam_tools
from rocit.constants import HUMAN_CHROMOSOME_ENUM
from make_sample_training_data import get_bam_path
from variant_loader import load_short_read_variants,load_long_read_variants,BASE_ENUM

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
FILTER_STATUS_ENUM = pl.Enum(['pass','fail','missing'])

def get_short_read_bam_path(sample_id:str):
    base_id = sample_id.split('_')[0]
    sample_type = 'T' if 'TU' in sample_id else 'N'
    cancer_type = 'prostate' if sample_id.startswith('BS') else 'ovarian'
    
    base_dir = Path(f'/hot/user/nmatulionis/short_read/output/{cancer_type}')
    
    sample_dir = base_dir/f'{base_id}/{base_id}/alignments/dna'
    
    filename = f'{base_id}-{sample_type}.redux.bam'
    filepath = sample_dir/filename

    return filepath



def get_variant_reads_df(long_read_variants, sample_bam_path, max_workers=None):
    schema = {
    "read_index": pl.Categorical,
    "contains_snv": pl.Boolean,
    "chromosome":HUMAN_CHROMOSOME_ENUM,
    "position": pl.Int32,
    "ref": BASE_ENUM,
    "alt": BASE_ENUM,
    "filter": pl.Categorical,
    "tumor_ref_count": pl.Int16,
    "tumor_alt_count": pl.Int16,
    'tumor_ref_count_sage':pl.Int16,
    'tumor_alt_count_sage':pl.Int16,
    "SAGE_filter_status": FILTER_STATUS_ENUM
    }   
    rows = list(long_read_variants.iter_rows(named=True))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers,mp_context=ctx) as executor:
        results = executor.map(bam_tools.get_variant_reads, rows, [sample_bam_path] * len(rows))
    results = [df.cast(schema) for df in results if not df is None and len(df)>0]

   
    df= pl.concat(list(results))
    
    #df = df.with_columns(pl.col('ref').cast(BASE_ENUM),pl.col('alt').cast(BASE_ENUM),pl.col('read_index').cast(pl.Categorical))
    #df = df.with_columns(pl.col('tumor_ref_count').cast(pl.Int16),pl.col('tumor_alt_count').cast(pl.Int16),pl.col('position').cast(pl.Int32))
    return df


def load_short_read_wrapper(sample_id:str):
    short_read = load_short_read_variants(sample_id,pass_filter=False)
    short_read = short_read.with_columns(
    pl.when(pl.col("filter") == "PASS")
    .then(pl.lit("pass"))
    .otherwise(pl.lit("fail"))
    .alias("SAGE_filter_status")
    )
    return short_read

def get_long_read_variants_with_short_read_status(sample_id:str):
    long_read = load_long_read_variants(sample_id,pass_filter=True)
    long_read = long_read.select(['chromosome','position','ref','alt','tumor_ref_count','tumor_alt_count'])
    short_read = load_short_read_wrapper(sample_id).rename({'tumor_ref_count':'tumor_ref_count_sage','tumor_alt_count':'tumor_alt_count_sage'})
    
    
    long_read = long_read.join(short_read,how='left',on=['chromosome','position','ref','alt'],coalesce=True)
    long_read = long_read.with_columns(pl.col('SAGE_filter_status').fill_null('missing').cast(FILTER_STATUS_ENUM))
    return long_read


def get_short_read_variants(sample_id:str):

    short_read = load_short_read_wrapper(sample_id)
    short_read = short_read.with_columns(pl.col('SAGE_filter_status').cast(FILTER_STATUS_ENUM),pl.col('filter').cast(pl.Categorical))
    return short_read


def normal_to_tumor_id(sample_id:str):
    base_id = sample_id.split('_')[0]
    return f'{base_id}_NL'
if __name__ =='__main__':
    #sample_ids = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    sample_ids = ['216_NL','244_NL','264_NL','053_NL','BS14772_NL','BS15145_NL']
    sample_id = sample_ids[int(sys.argv[1])]
    variant_id = normal_to_tumor_id(sample_id) if 'NL' in sample_id else sample_id
    variant_mode = sys.argv[2]
    read_type = sys.argv[3]

    if read_type =='long_read':
        sample_bam_path = get_bam_path(sample_id)
    elif read_type =='short_read':
        sample_bam_path = get_short_read_bam_path(sample_id)

    out_dir = Path(f'/hot/user/tobybaker/ROCIT_Paper/output/read_variant_store/variant_{variant_mode}_bam_{read_type}')
    out_dir.mkdir(exist_ok=True)
    if variant_mode =='long_read':

        
        long_read_variants = get_long_read_variants_with_short_read_status(variant_id)
       
        
        read_df = get_variant_reads_df(long_read_variants,sample_bam_path)
        
        out_path = out_dir/f'{sample_id}_reads.parquet'
        read_df.write_parquet(out_path)
    if variant_mode =='short_read':
        
        
        short_read_variants = get_short_read_variants(variant_id)
        
  
        
        read_df = get_variant_reads_df(short_read_variants,sample_bam_path)
        read_df = read_df.with_columns(pl.col('SAGE_filter_status').cast(FILTER_STATUS_ENUM),pl.col('filter').cast(pl.Categorical))
        
        out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/output/short_read_variants')
        out_path = out_dir/f'{sample_id}_reads.parquet'
        read_df.write_parquet(out_path)