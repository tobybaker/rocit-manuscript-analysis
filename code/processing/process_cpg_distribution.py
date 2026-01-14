import polars as pl
import numpy as np
from pathlib import Path
METHYLATION_SCALE = 256.0
def get_aggregate_methylation_distribution(methylation_df,min_n_cpgs:int=10):

    percentiles = np.arange(5,100,5)/100.0
    methylation_df = methylation_df.with_columns(
        pl.col("methylation").cast(pl.Float32)/METHYLATION_SCALE + 0.5/METHYLATION_SCALE
    )
    aggregate_methylation_df = methylation_df.group_by(['chromosome','position']).agg(
    pl.when(pl.col("methylation").count() >= min_n_cpgs)
      .then(pl.col("methylation").quantile(p).cast(pl.Float32))
      .otherwise(None)
      .alias(f"methylation_percentile_{int(p * 100)}")
    for p in percentiles
    ).sort(['chromosome','position'])

    return aggregate_methylation_df

def process_default_methylation_df(methylation_df):
    methylation_df = methylation_df.drop_nulls("position")
    methylation_df = methylation_df.with_columns((pl.col('methylation').cast(pl.Float64)/256.0 +0.5/256.0).alias('methylation'))
    return methylation_df
if __name__ =='__main__':
    
    in_dir = Path('../../../ROCIT_Non_Paper/data/extracted_cpg/293T/')
    in_path = in_dir/'293T_chr10_cpg_methylation.parquet'

    in_df = pl.read_parquet(in_path)
    

    
    result = in_df.group_by(['chromosome','position']).agg(
    pl.col("methylation_prob").quantile(p).alias(f"methylation_percentile_{int(p * 100)}") 
    for p in percentiles
    ).sort(['chromosome','position'])

    min_n_cpgs=10
    
    
    
    print(result)
