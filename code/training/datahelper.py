import polars
from pathlib import Path
import sys
import datahelper
import rocit
from rocit.data import EmbeddingStore,ReadDataset
from torch.utils.data import Dataset, DataLoader
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
def read_parquet(filepath:str):
    df = polars.read_parquet(filepath)
    df = df.drop([c for c in df.columns if c.startswith("__index_level_")])
    return df
def load_read_data(filepath:str):
    
    df = read_parquet(filepath)
    
    df = df.with_columns(
    polars.col("Chromosome").cast(polars.Enum(CHROMOSOMES)),
    polars.col("Read_Position").cast(polars.Int32),
    polars.col("Methylation").cast(polars.UInt8)
    )

    if 'Tumor_Read' in df.columns:
        df = df.with_columns(polars.col("Tumor_Read").cast(polars.Float32))
    if 'Strand' in df.columns:
        df = df.with_columns(polars.col("Strand").cast(polars.Enum(["+","-"])))
        
    return df
def process_grouped_read_data(read_data,sample_id=None):
    if not sample_id is None:
        read_data = read_data.with_columns(polars.lit(sample_id).alias("Sample_ID"))
    
    read_data = read_data.with_columns(
    polars.col("Read_Index").cast(polars.Categorical),
    polars.col("Sample_ID").cast(polars.Categorical),
    )
    
    read_data = read_data.filter(~polars.col("Supplementary_Alignment"))
   
    read_data = read_data.unique(subset=['Read_Index','Read_Position'])
    return read_data

def get_sample_training_reads(sample_id:str):
    base_dir =Path('/hot/user/tobybaker/CellTypeClassifier/data')
    if 'NL' in sample_id:
        data_dir = base_dir/'labelled_read_dfs_normal_all_positions'
    else:
        data_dir = base_dir/'labelled_read_dfs_qc_ASCAT_all_positions'
    df_store = []
    for filepath in data_dir.glob(f'{sample_id}_labelled_reads.parquet'):
        df = load_read_data(filepath)
        df_store.append(df)
    read_data = polars.concat(df_store)
    read_data = process_grouped_read_data(read_data,sample_id)

    return read_data

def get_sample_inference_reads(sample_id:str):
    base_dir =Path('/hot/user/tobybaker/CellTypeClassifier/data/processed_methylation_info_redo/tumor/')
    data_dir = base_dir/ sample_id
    
    df_store = []
    for filepath in data_dir.glob(f'cpg_methylation_data_*.parquet'):
        df = load_read_data(filepath)
        
        df_store.append(df)

    read_data = polars.concat(df_store)
    
    read_data = process_grouped_read_data(read_data,sample_id)
    
    return read_data

def get_sample_inference_store(sample_id):
    
    sample_dist_df = load_sample_dist_df(sample_id)
    cell_map_df = load_cell_map_df()

    
    sample_source = EmbeddingStore('Sample_Distribution',sample_dist_df,['Sample_ID','Chromosome','Position'])
    cell_map_source = EmbeddingStore('Cell_Map',cell_map_df,['Chromosome','Position'])
    
    embedding_sources = {sample_source.name:sample_source,cell_map_source.name:cell_map_source}

    read_data = get_sample_inference_reads(sample_id)
    

    label_cols = ['Sample_ID','Read_Index','Chromosome']
    key_cols = ['Read_Index']
    
    inference_dataset = ReadDataset(read_data,label_cols,key_cols,embedding_sources)

    
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

    label_cols = ['Read_Index','Chromosome','Tumor_Read']
    key_cols = ['Read_Index']
    
    train_read_data = read_data.filter(polars.col("Chromosome").str.contains_any(train_chromosomes))
    test_read_data = read_data.filter(polars.col("Chromosome").str.contains_any(test_chromosomes))
    val_read_data = read_data.filter(polars.col("Chromosome").str.contains_any(val_chromosomes))

    train_dataset = ReadDataset(train_read_data,label_cols,key_cols,embedding_sources)
    test_dataset = ReadDataset(test_read_data,label_cols,key_cols,embedding_sources)
    val_dataset = ReadDataset(val_read_data,label_cols,key_cols,embedding_sources)
    
    return rocit.ROCITTrainStore(train_dataset,test_dataset,val_dataset,embedding_sources)



