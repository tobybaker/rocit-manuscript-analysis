import polars
import numpy as np
from pathlib import Path
import sys
import datahelper
import rocit
import itertools
from rocit.data import EmbeddingStore,ReadDatasetBuilder
from torch.utils.data import Dataset, DataLoader

def get_run_params(param_config):
    param_names = list(param_config.keys())
    param_values = list(param_config.values())
    
    return [
        dict(zip(param_names, combination))
        for combination in itertools.product(*param_values)
    ]
CHROMOSOMES  = [f'chr{x}' for x in range(1,23)] +['chrX','chrY','chrM']
def tumor_to_normal_id(tumor_id):
    if not 'TU' in tumor_id:
        raise ValueError(f'{tumor_id} is not a tumor id.')
    return tumor_id.replace('TU','NL')

def normal_to_tumor_id(normal_id):
    if not 'NL' in normal_id:
        raise ValueError(f'{normal_id} is not a normal id.')
    return normal_id.replace('NL','TU')
def load_cell_map_df():
    base_dir = Path('/hot/user/tobybaker/CellTypeClassifier/data')
    cell_type_path = base_dir/'complete_cell_type_methylation_with_average_only.parquet'
    cell_map_df = read_parquet(cell_type_path)
    
    cell_map_df = cell_map_df.with_columns(
    polars.col("Chromosome").cast(polars.Enum(CHROMOSOMES)),
    )
    return cell_map_df
    
def load_sample_dist_df(sample_id):
    base_dir = Path('/hot/user/tobybaker/CellTypeClassifier/data/processed_methylation_distributions/tumor')

    sample_dist_path = base_dir/f'{sample_id}/combined_distribution.parquet'
    sample_dist_df = read_parquet(sample_dist_path)
    
    #sample_dist_df = sample_dist_df.with_columns(polars.lit(sample_id).alias("Sample_ID"))

    sample_dist_df = sample_dist_df.with_columns(
    polars.col("Chromosome").cast(polars.Enum(CHROMOSOMES)),
    #polars.col("Sample_ID").cast(polars.Categorical)
    )

    return sample_dist_df
        
#hack for pandas
def read_parquet(filepath:str,scan:bool=False):
    if scan:
        df = polars.scan_parquet(filepath)
    else:
        df = polars.read_parquet(filepath)
    df = df.drop([c for c in df.columns if c.startswith("__index_level_")])
    return df

def inspect_memory(df: polars.DataFrame) -> None:
    """
    Prints memory usage per column (descending).
    Truncates verbose Enum/Categorical type descriptions.
    """
    total_mb = df.estimated_size() / (1024**2)
    print(f"--- Total Memory: {total_mb:.2f} MB ---")
    print(f"{'Column':<30} | {'Type':<15} | {'Size (MB)':>10}")
    print("-" * 61)

    stats = []
    for col in df.columns:
        dtype = df[col].dtype
        size_mb = df[col].estimated_size() / (1024**2)
        
        # Clean up ugly Enum/Categorical print output
        if isinstance(dtype, polars.Enum):
            type_str = "Enum"
        elif isinstance(dtype, polars.Categorical):
            type_str = "Categorical"
        else:
            type_str = str(dtype)
            
        stats.append((col, type_str, size_mb))

    # Sort by size descending
    stats.sort(key=lambda x: x[2], reverse=True)

    for col, type_str, size_mb in stats:
        print(f"{col:<30} | {type_str:<15} | {size_mb:>10.2f}")
def load_read_data(filepath:str,scan=False):
    
    df = read_parquet(filepath,scan=scan)
    
    df = df.with_columns(
    polars.col("Chromosome").cast(polars.Enum(CHROMOSOMES)),
    polars.col("Methylation").cast(polars.UInt8),
    polars.col("Read_Index").cast(polars.Categorical)
    )
    
    if 'Tumor_Read' in df.columns:
        df = df.with_columns(polars.col("Tumor_Read").cast(polars.Boolean))
    if 'Strand' in df.columns:
        df = df.drop('Strand')
    if 'Read_Count' in df.columns:
        df = df.drop('Read_Count')
    df = df.filter(~polars.col("Supplementary_Alignment"))
   
    df = df.unique(subset=['Read_Index','Read_Position'])
    return df

def get_sample_training_reads(sample_id:str):
    base_dir =Path('/hot/user/tobybaker/CellTypeClassifier/data')
    if 'NL' in sample_id:
        data_dir = base_dir/'labelled_read_dfs_normal_all_positions'
    else:
        data_dir = base_dir/'labelled_read_dfs_qc_ASCAT_all_positions'
    df_store = []
    with polars.StringCache():
        for filepath in data_dir.glob(f'{sample_id}_labelled_reads*.parquet'):
            df = load_read_data(filepath)

            df = df.with_columns(
                polars.lit(sample_id).cast(polars.Categorical).alias("Sample_ID")
            )
            
            df_store.append(df)
    read_data = polars.concat(df_store)
    
    return read_data

def get_sample_inference_reads(sample_id:str):
    base_dir =Path('/hot/user/tobybaker/CellTypeClassifier/data/processed_methylation_info_redo/tumor/')
    data_dir = base_dir/ sample_id
    
    read_store = []
    with polars.StringCache():
        for filepath in data_dir.glob(f'cpg_methylation_data_*.parquet'):
            df = load_read_data(filepath,scan=True)
            df = df.with_columns(
                polars.lit(sample_id).cast(polars.Categorical).alias("Sample_ID")
            )

            read_store.append(df)
            
    return read_store

def get_sample_inference_store(sample_id):
    
    sample_dist_df = load_sample_dist_df(sample_id)
    cell_map_df = load_cell_map_df()
    
    sample_source = EmbeddingStore('Sample_Distribution',sample_dist_df,['Chromosome','Position'])
    cell_map_source = EmbeddingStore('Cell_Map',cell_map_df,['Chromosome','Position'])
    
    embedding_sources = {sample_source.name:sample_source,cell_map_source.name:cell_map_source}

    read_store = get_sample_inference_reads(sample_id)

    label_cols = ['Sample_ID','Read_Index','Chromosome']
    key_cols = ['Read_Index']
    
    inference_dataset_builder = ReadDatasetBuilder(read_store,label_cols,key_cols,embedding_sources)
    inference_dataset = inference_dataset_builder.build()
    
    return rocit.ROCITInferenceStore(inference_dataset,embedding_sources)

def get_sample_train_datasets(sample_id,add_normal=False):
    all_chromosomes = [f'chr{x}' for x in range(1,23)] +['chrX']
    test_chromosomes = ['chr4','chr21']
    val_chromosomes = ['chr5','chr22']
    non_train_chromosomes = set(test_chromosomes) | set(val_chromosomes)

    # Filter the list
    train_chromosomes = [chrom for chrom in all_chromosomes if chrom not in non_train_chromosomes]

    sample_dist_df = load_sample_dist_df(sample_id)
    cell_map_df = load_cell_map_df()
    
    sample_source = EmbeddingStore('Sample_Distribution',sample_dist_df,['Chromosome','Position'])
    cell_map_source = EmbeddingStore('Cell_Map',cell_map_df,['Chromosome','Position'])
    
    embedding_sources = {sample_source.name:sample_source,cell_map_source.name:cell_map_source}

    read_data = get_sample_training_reads(sample_id)
    if add_normal:
        normal_sample_id = f'{sample_id.split("_")[0]}_NL'
        normal_read_data = get_sample_training_reads(normal_sample_id)
        read_data = polars.concat([read_data,normal_read_data])

    label_cols = ['Sample_ID','Read_Index','Chromosome','Tumor_Read']
    key_cols = ['Read_Index']
    
    train_read_data = read_data.filter(polars.col("Chromosome").is_in(train_chromosomes))
    test_read_data = read_data.filter(polars.col("Chromosome").is_in(test_chromosomes))
    val_read_data = read_data.filter(polars.col("Chromosome").is_in(val_chromosomes))


    train_dataset_builder = ReadDatasetBuilder(train_read_data,label_cols,key_cols,embedding_sources)
    test_dataset_builder = ReadDatasetBuilder(test_read_data,label_cols,key_cols,embedding_sources)
    val_dataset_builder = ReadDatasetBuilder(val_read_data,label_cols,key_cols,embedding_sources)
    
    return rocit.ROCITTrainStore(train_dataset_builder.build(),test_dataset_builder.build(),val_dataset_builder.build(),embedding_sources)

def load_read_extent_store(sample_id:str):
    read_extent_dir = Path('/hot/user/tobybaker/CellTypeClassifier/data/read_extent')
    read_extent_path = read_extent_dir/f'{sample_id}_read_extent.parquet'
    read_extent =read_parquet(read_extent_path)
    
    read_extent = read_extent.with_columns(
    polars.col("Chromosome").cast(polars.Enum(CHROMOSOMES)),
    polars.col("Read_Index").cast(polars.Categorical),
    polars.col("Reference_Start").cast(polars.Int32),
    polars.col("Reference_End").cast(polars.Int32)
    )
    read_extent = read_extent.drop(['Reference_Start','Reference_End'])
    return read_extent
def get_sample_train_length_datasets(sample_id,read_length:int,min_length:int=15000):
    RNG = np.random.default_rng(10125)
    
    all_chromosomes = [f'chr{x}' for x in range(1,23)] +['chrX']
    test_chromosomes = ['chr4','chr21']
    val_chromosomes = ['chr5','chr22']
    non_train_chromosomes = set(test_chromosomes) | set(val_chromosomes)

    # Filter the list
    train_chromosomes = [chrom for chrom in all_chromosomes if chrom not in non_train_chromosomes]

    sample_dist_df = load_sample_dist_df(sample_id)
    cell_map_df = load_cell_map_df()
    
    sample_source = EmbeddingStore('Sample_Distribution',sample_dist_df,['Chromosome','Position'])
    cell_map_source = EmbeddingStore('Cell_Map',cell_map_df,['Chromosome','Position'])
    
    embedding_sources = {sample_source.name:sample_source,cell_map_source.name:cell_map_source}

    read_data = get_sample_training_reads(sample_id)
    
    read_extent = load_read_extent_store(sample_id)

    read_extent = read_extent.filter(polars.col("Read_Length") >= min_length)
    read_extent = read_extent.with_columns(
    polars.Series("Sampled_Read_Start", RNG.integers(
        0, 
        read_extent["Read_Length"].to_numpy() -read_length
    ))
    )

    read_extent = read_extent.with_columns((polars.col('Sampled_Read_Start')+read_length).alias('Sampled_Read_End'))
    
    
    read_data = read_data.join(read_extent,how='inner',on=['Read_Index','Chromosome'])
    
    read_data = read_data.filter(polars.col("Read_Position").is_between(polars.col("Sampled_Read_Start"), polars.col("Sampled_Read_End"), closed="both"))
    read_data = read_data.drop(['Read_Length','Sampled_Read_Start','Sampled_Read_End'])
    
    label_cols = ['Read_Index','Chromosome','Tumor_Read']
    key_cols = ['Read_Index']
    
    train_read_data = read_data.filter(polars.col("Chromosome").is_in(train_chromosomes))
    test_read_data = read_data.filter(polars.col("Chromosome").is_in(test_chromosomes))
    val_read_data = read_data.filter(polars.col("Chromosome").is_in(val_chromosomes))

    train_dataset_builder = ReadDatasetBuilder(train_read_data,label_cols,key_cols,embedding_sources)
    test_dataset_builder = ReadDatasetBuilder(test_read_data,label_cols,key_cols,embedding_sources)
    val_dataset_builder = ReadDatasetBuilder(val_read_data,label_cols,key_cols,embedding_sources)
    
    return rocit.ROCITTrainStore(train_dataset_builder.build(),test_dataset_builder.build(),val_dataset_builder.build(),embedding_sources)

