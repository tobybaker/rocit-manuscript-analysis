import polars as pl
import numpy as np
import itertools
from pathlib import Path
def load_read_labels(sample_id:str,cut_off:float=0.0):
    if not 'TU' in sample_id:
        raise ValueError(f'Unable to load read labels for {sample_id}')
    #053_TU_add_normal_False/full_datasets/
    in_dir = Path(f'/hot/user/tobybaker/ROCIT_Paper/predictions/main_predictions/{sample_id}_add_normal_False/full_datasets')

    label_path = in_dir/f'train_{sample_id}_add_normal_False_out_{sample_id}_all_reads.parquet'
    
    #label_path = f'/scratch/transformer_preds_main_ASCAT/transformer_preds_out_{sample_id}_model_{sample_id}_add_normal_False.tsv'
    read_labels = pl.scan_parquet(label_path)
    read_labels = read_labels.with_columns((pl.col('tumor_probability')>0.5).alias('tumor_read'))
    read_labels = read_labels.with_columns(pl.col('chromosome').cast(pl.Categorical),pl.col('read_index').cast(pl.Categorical))

    return read_labels


def load_haplotags(sample_id):
    sample_base,sample_type = sample_id.split('_')
    PHASING_DIR =  Path('/hot/user/datkinson/merged_phasing/03_PHASING/TOPMED_001')
    #PHASING_DIR = '/scratch/HapTags'
    if 'BS' in sample_id:
        hap_path = PHASING_DIR/f'{sample_base.replace("BS","")}.PASS_TOPMED001.HapTags.tsv'
    else:
        hap_path = PHASING_DIR/f'{sample_base}.PASS_TOPMED001.HapTags.tsv'
 
    haplotags = pl.scan_csv(hap_path,separator="\t").select(['chrom','read_name','haplotag'])
    haplotags = haplotags.rename({'read_name':'read_index','chrom':'chromosome','haplotag':'haplotag'})
    haplotags = haplotags.with_columns(pl.col('chromosome').cast(pl.Categorical),pl.col('read_index').cast(pl.Categorical))
    return haplotags


def load_read_methylation(methylation_data_path:str):

    """
    Load the read methylation data from path
    """
    
    read_methylation_data = pl.scan_parquet(methylation_data_path)

    
    read_methylation_data = read_methylation_data.filter(~pl.col('supplementary_alignment')).drop_nulls()

    
    read_methylation_data = read_methylation_data.with_columns((pl.col('methylation')/256.0 + 0.5/256.0).alias('methylation'))

    
    return read_methylation_data

def load_methylation_data(sample_id:str,chromosome:str):

    in_dir = Path(f'/hot/user/tobybaker/ROCIT_Paper/input_data/cpg_methylation/{sample_id}')
    methylation_data_path = in_dir/f'{sample_id}_{chromosome}_cpg_methylation.parquet'
    read_methylation_data = load_read_methylation(methylation_data_path)

    haplotags = load_haplotags(sample_id)
    
    
    read_methylation_data = read_methylation_data.join(haplotags,how='left',on=['chromosome','read_index'])
    

    read_labels = load_read_labels(sample_id)
    read_methylation_data = read_methylation_data.join(read_labels,how='left',on=['chromosome','read_index'])
    
    return read_methylation_data.collect()



def get_run_params(param_config):


    param_names = list(param_config.keys())
    param_values = list(param_config.values())

    return [
        dict(zip(param_names, combination))
        for combination in itertools.product(*param_values)
    ]  

def load_fire_calls(sample_id,chromosome):

    in_dir = Path(f'/hot/user/candrasz/read-classification-fiberseq/data-processing/FIRE-processing/out/results-BULK/{sample_id}_TU-BULK/additional-outputs-v0.1/fire-peaks')
    fire_path = in_dir/f'{sample_id}_TU-BULK-v0.1-fire-elements.bed.gz'

    col_names = ['chromosome','start','end','read_index','count','strand','start2','end2','data','score','name']
    fire_data = pl.scan_csv(fire_path,separator="\t",has_header=False,new_columns=col_names)
    fire_data = fire_data.with_columns(pl.col('chromosome').cast(pl.Categorical),pl.col('read_index').cast(pl.Categorical))
    fire_data = fire_data.filter(pl.col('chromosome')==chromosome)
    return fire_data

def get_fire_data_with_region(fire_data,region,window_buffer,margin=0):
    buffer_reduced= window_buffer-margin
    fire_regions = fire_data.filter(pl.col('start').is_between(region['start']-buffer_reduced,region['end']+buffer_reduced))
    fire_regions = fire_regions.filter(pl.col('end').is_between(region['start']-buffer_reduced,region['end']+buffer_reduced))
    return fire_regions.collect()


