import polars as pl

from pathlib import Path



def get_phasing_filepath(sample_id,mode):
    phasing_dir =  Path('/hot/user/datkinson/merged_phasing/03_PHASING/')
    sample_base,sample_type = sample_id.split('_')
    
    if 'BS' in sample_base:
        sample_base = sample_base.replace('BS','')
    
    if mode =='haploblocks':
        suffix = '.PASS_dbSNP.blocks.tsv'
    if mode =='haplotags':
        suffix = '.PASS_dbSNP.HapTags.tsv'
    filepath = phasing_dir/f'{sample_base}{suffix}'
    return filepath

def load_haploblocks(sample_id):
    filepath = get_phasing_filepath(sample_id, 'haploblocks')
    
    haploblocks = pl.read_csv(
        filepath, 
        separator="\t", 
        columns=['phase_block_id', 'chrom', 'start', 'end', 'num_variants']
    )
    
    haploblocks = (
        haploblocks
        .rename({
            'phase_block_id': 'block_id', 
            'chrom': 'chromosome', 
            'start': 'block_start', 
            'end': 'block_end', 
            'num_variants': 'n_variants'
        })
        .with_columns(
            (pl.col('block_end') - pl.col('block_start')).alias('block_size').cast(pl.Int32),
            pl.col('chromosome').cast(pl.Categorical),
            pl.col('block_start').cast(pl.Int32),
            pl.col('block_end').cast(pl.Int32),
            pl.col('n_variants').cast(pl.Int32)
        )
    )
    return haploblocks

def load_haplotags(sample_id):
    filepath = get_phasing_filepath(sample_id, 'haplotags')

    haplotags = pl.read_csv(
        filepath, 
        separator="\t", 
        columns=['chrom', 'read_name', 'haplotag', 'phase_block_id']
    )
    haplotags = haplotags.rename({
        'read_name': 'read_index', 
        'phase_block_id': 'block_id',
        'chrom': 'chromosome',
        'haplotag': 'haplotag'
    })

    # Group by chromosome and read_index (updated to snake_case keys)
    counts = haplotags.group_by(['chromosome', 'read_index']).len()
    
    # Filter for reads appearing exactly once
    valid_reads = counts.filter(pl.col('len') == 1).select('read_index')

    # Join back to filter the original dataframe using the snake_case key
    haplotags = haplotags.join(valid_reads, on='read_index', how='semi').sort('read_index')

    haplotags = haplotags.with_columns(
            pl.col('chromosome').cast(pl.Categorical),
            pl.col('haplotag').cast(pl.Int8)
        )
    return haplotags