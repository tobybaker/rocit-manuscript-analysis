import sys
import polars as pl
import numpy as np
from pathlib import Path

CHROMOSOMES  = [f'chr{x}' for x in range(1,23)] +['chrX','chrY','chrM']
METHYLATION_SCALE = 256

def load_sample_dist_df(sample_id:str):
    base_dir = Path('/hot/user/tobybaker/CellTypeClassifier/data/processed_methylation_distributions/tumor')

    sample_dist_path = base_dir/f'{sample_id}/combined_distribution.parquet'
    sample_dist_df = pl.scan_parquet(sample_dist_path)
    
    #sample_dist_df = sample_dist_df.with_columns(polars.lit(sample_id).alias("sample_id"))

    sample_dist_df = sample_dist_df.with_columns(
    pl.col("chromosome").cast(pl.Enum(CHROMOSOMES)),
    #polars.col("sample_id").cast(polars.Categorical)
    )
    sample_dist_df = sample_dist_df.drop_nulls()
    
    return sample_dist_df
        
def get_aggregate_read_distribution(methylation_df):

    percentiles = np.arange(5,100,5)/100.0
    methylation_df = methylation_df.with_columns(
        
    )
    aggregate_methylation_df = methylation_df.group_by(['chromosome','read_index']).agg((pl.col("methylation").quantile(p).cast(pl.Float32).alias(f"methylation_percentile_{int(p * 100)}") for p in percentiles)).sort(['chromosome','read_index'])
    n_cpgs =  methylation_df.group_by(['chromosome','read_index']).len().rename({'len':'n_cpgs'})
    aggregate_methylation_df = aggregate_methylation_df.join(n_cpgs,how='inner',coalesce=True,on=['chromosome','read_index'])
    return aggregate_methylation_df
def load_cell_map_df():
    base_dir = Path('/hot/user/tobybaker/CellTypeClassifier/data')
    cell_type_path = base_dir/'complete_cell_type_methylation_with_average_only.parquet'
    cell_map_df = pl.scan_parquet(cell_type_path)
    
    cell_map_df = cell_map_df.with_columns(
    pl.col("chromosome").cast(pl.Enum(CHROMOSOMES)),
    )

    return cell_map_df
def load_sample_labelled_reads(sample_id:str):
    in_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/labelled_data')
    filename = f'{sample_id}_labelled_data.parquet'

    filepath = in_dir/filename
    in_df =  pl.scan_parquet(filepath)
    in_df = in_df.with_columns(
    pl.col("chromosome").cast(pl.Enum(CHROMOSOMES)),
    pl.col("methylation").cast(pl.Float32)/METHYLATION_SCALE + 0.5/METHYLATION_SCALE
    )
    return in_df
def get_relative_distribution(sample_labelled_reads,sample_dist):
    merged = sample_labelled_reads.join(sample_dist,how='inner',on=['chromosome','position'])

    ref_cols = [c for c in sample_dist.collect_schema().names() if c.startswith("methylation_percentile_")]
    merged = merged.with_columns(
    (
        pl.sum_horizontal(*(pl.col(c) < pl.col("methylation") for c in ref_cols))
        / len(ref_cols)
    ).alias("rel_rank")
    )
    percentiles = np.arange(5,100,5)/100.0
    aggregate_df = merged.group_by(['chromosome','read_index']).agg((pl.col("rel_rank").quantile(p).cast(pl.Float32).alias(f"rank_percentile_{int(p * 100)}") for p in percentiles)).sort(['chromosome','read_index'])
    return aggregate_df
def get_aggregate_cell_map_distribution(sample_labelled_reads,cell_map,methylation_col_start='average_methylation_'):
    avg_cols = [c for c in cell_map.collect_schema().names() if c.startswith(methylation_col_start)]
    merged = sample_labelled_reads.join(cell_map,how='inner',on=['chromosome','position'])
    corr_exprs = [
    pl.corr("methylation", c, method="pearson").alias(f"corr_{c.replace(methylation_col_start,'')}")
    for c in avg_cols
    ]

    df = merged.group_by(['chromosome','read_index']).agg(corr_exprs)
    return df

def get_training_data(sample_id):
    sample_labelled_reads = load_sample_labelled_reads(sample_id)
    cell_map = load_cell_map_df()
    sample_dist = load_sample_dist_df(sample_id)
    
    read_distribution = get_aggregate_read_distribution(sample_labelled_reads)
    cell_map_distribution = get_aggregate_cell_map_distribution(sample_labelled_reads,cell_map)
    relative_distribution = get_relative_distribution(sample_labelled_reads,sample_dist)
    
    overall_distribution = read_distribution.join(cell_map_distribution,how='full',on=['chromosome','read_index'],coalesce=True)
    overall_distribution = overall_distribution.join(relative_distribution,how='full',on=['chromosome','read_index'],coalesce=True)
    return overall_distribution

