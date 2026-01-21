import polars as pl

import cn_loader
import cluster_loader
import phasing_loader
import variant_loader
from pathlib import Path
from rocit.preprocessing import train_data

def get_short_read_variant_filter_type(sample_id:str):
    if sample_id.startswith('BS'):
        return 'Remove_Germline'
    return 'PASS_Intersection'

def run_short_read_filtering(sample_id,long_read_variants,short_read_variants):
    #doing a simple germline filter
    filter_type = get_short_read_variant_filter_type(sample_id)
    
    if filter_type=='Remove_Germline':
        germline_filter = short_read_variants.filter(pl.col('filter').str.contains('Germline'))
        long_read_variants = long_read_variants.join(germline_filter,how='anti',on=['chromosome','position'])
       
    else:
        short_read_pass = short_read_variants.filter(pl.col('filter') =='PASS').drop('filter')
        long_read_variants = long_read_variants.join(short_read_pass,how='inner',on=['chromosome','position','ref','alt'])
    return long_read_variants

def get_bam_path(sample_id:str):
    in_dir = Path('/hot/user/candrasz/no-kinetic-bams/')
    return in_dir/f'{sample_id}.HT_split.primary_nokinetics.bam'
if __name__ =='__main__':
    sample_id = '244_TU'
    '''sample_cn = cn_loader.load_cn(sample_id)
    
    cluster_labels = cluster_loader.load_cluster_labels(sample_id)
    snv_cluster_assignments = cluster_loader.load_cluster_assignments(sample_id,cluster_labels)
    
    haplotags = phasing_loader.load_haplotags(sample_id)
    haploblocks = phasing_loader.load_haploblocks(sample_id)
    

    long_read_variants = variant_loader.load_long_read_variants(sample_id)
    short_read_variants = variant_loader.load_short_read_variants(sample_id)

    long_read_variants = run_short_read_filtering(sample_id,long_read_variants,short_read_variants)
    sample_bam_path = get_bam_path(sample_id)
    pretrain_data = train_data.ROCITPreTrainData(sample_id,sample_bam_path,sample_cn,long_read_variants,haplotags,haploblocks,cluster_labels,snv_cluster_assignments)
    pretrain_data.save('/scratch/temp.pkl')'''
    pretrain_data = train_data.ROCITPreTrainData.load('/scratch/temp.pkl')
    train_data.make_read_labels(pretrain_data)
    