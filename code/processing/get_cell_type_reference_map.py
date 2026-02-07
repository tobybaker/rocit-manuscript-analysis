
import pyBigWig
import numpy as np
import polars as pl
import os
import re

from pathlib import Path
from functools import reduce
from tqdm import tqdm
CHROMOSOMES = [f'chr{x}' for x in range(1,23)] + ['chrX']
def load_bigwig_paths(data_dir):
    
    bigwig_paths = {}
    for filepath in data_dir.glob('*.hg38.bigwig'):
        #custom match function to find cell type in filename
        match = re.search(r"_(.*?)-(Z0|11)", filepath.name)

        if match:
            cell_type = match.group(1)
        else:
            raise ValueError(f'no cell type for {filepath}')
        
        if not cell_type in bigwig_paths:
            bigwig_paths[cell_type] = []
        bigwig_paths[cell_type].append(filepath)
    return bigwig_paths

def load_big_wig_df(bigwig_path,eps=1e-9):
    bw = pyBigWig.open(str(bigwig_path))

    data = []
    for chromosome in bw.chroms():
        if not chromosome in CHROMOSOMES:
            continue
        intervals = bw.intervals(chromosome)
        if intervals:  # Some chromosomes may have no data
            for start, end, value in intervals:
                if value < -eps:
                    continue
                data.append((chromosome, start, value))
    
    bw.close()
    
    # Construct Polars DataFrame
    df = pl.DataFrame(
        data,
        schema={
            "chromosome": pl.Enum(CHROMOSOMES),
            "position": pl.UInt32,
            "average_methylation": pl.Float32
        },
        orient="row"
    )
    
    return df


def get_cell_type_df(bigwig_paths):
    all_dfs = []
    for bigwig_path in bigwig_paths:
        big_wig_df = load_big_wig_df(bigwig_path)
        all_dfs.append(big_wig_df)
        
    all_dfs = pl.concat(all_dfs)
    df_agg = all_dfs.group_by(['chromosome','position']).agg(pl.col('average_methylation').mean())
    return df_agg

def join_frames(df_store):
    return reduce(
        lambda left, right: left.join(
            right,
            on=["chromosome", "position"],
            how="full",
            coalesce=True,
        ),
        df_store,
    )
def get_combined_cell_type_df(all_bigwig_paths):
    df_store = []
    
    for cell_type,cell_type_bigwig_paths in tqdm(all_bigwig_paths.items(),desc='Loading cell types'):
        cell_type_df = get_cell_type_df(cell_type_bigwig_paths)
        cell_type_df = cell_type_df.rename({'average_methylation':f'average_methylation_{cell_type.lower()}'})
        df_store.append(cell_type_df.lazy())
        
    combined_df = join_frames(df_store).collect()
    combined_df = combined_df.sort(['chromosome','position'])
    return combined_df

if __name__ =='__main__':
    #downloaded from https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE186458&format=file
    data_dir = Path('/hot/user/tobybaker/CellTypeClassifier/data/GSMStore')
    all_bigwig_paths = load_bigwig_paths(data_dir)
    combined_cell_type_df = get_combined_cell_type_df(all_bigwig_paths)
    
    out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data')
    out_path = out_dir/'cell_type_average_methylation_atlas.parquet'
    combined_cell_type_df.write_parquet(combined_cell_type_df)