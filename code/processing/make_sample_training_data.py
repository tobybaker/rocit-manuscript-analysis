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
        germline_filter = short_read_variants.filter(pl.col('Filter').str.contains('Germline'))
        long_read_variants = long_read_variants.join(germline_filter,how='anti',on=['Chromosome','Position'])
       
    else:
        short_read_pass = short_read_variants.filter(pl.col('Filter') =='PASS').drop('Filter')
        long_read_variants = long_read_variants.join(short_read_pass,how='inner',on=['Chromosome','Position'])
    return long_read_variants

if __name__ =='__main__':
    sample_id = 'BS14772_TU'
    sample_cn = cn_loader.load_cn(sample_id)
    
    cluster_labels = cluster_loader.load_cluster_labels(sample_id)
    snv_cluster_assignments = cluster_loader.load_cluster_assignments(sample_id,cluster_labels)
    
    #haplotags = phasing_loader.load_haplotags(sample_id)
    #haploblocks = phasing_loader.load_haploblocks(sample_id)

    long_read_variants = variant_loader.load_long_read_variants(sample_id)
    short_read_variants = variant_loader.load_short_read_variants(sample_id)

    long_read_variants = run_short_read_filtering(sample_id,long_read_variants,short_read_variants)
    