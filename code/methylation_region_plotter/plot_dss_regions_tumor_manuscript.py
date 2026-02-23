import os
import sys
import plot_helper
import pandas as pd
import polars as pl
import numpy as np
from methylation_region_plotter import plot_region
from gene_caller import get_gene_data
from pathlib import Path


def get_params():
    sample_ids = ['053','216','244','264','BS14772','BS15145']
    chromosomes = [f'chr{x}' for x in range(1,23)]

    params = []
    for sample_id in sample_ids:
        for chromosome in chromosomes:
            param = {'Sample_ID':sample_id,'Chromosome':chromosome}
            params.append(param)
    return params


def load_target_gene_data():
    chromosome = 'chr20'
    sample_id = '244'
    region_data1 = {'sample_id':sample_id,'chromosome':chromosome,'start':43189750,'end':43204200,'gene':'PTPRT-DT'}
    region_data2 = {'sample_id':sample_id,'chromosome':chromosome,'start':43171800,'end':43188300,'gene':'PTPRT'}
    return pl.DataFrame([region_data1,region_data2])

def load_methylation_data(target_gene_df,buffer=10000):
    all_methylation_data = []
    for row in target_gene_df.iter_rows(named=True):

        tumor_id = f'{row["sample_id"]}_TU'
        normal_id = f'{row["sample_id"]}_NL'

        methylation_data = plot_helper.load_methylation_data(tumor_id,row["chromosome"])
        methylation_data = methylation_data.with_columns(pl.lit(row['sample_id']).alias('sample_id'),pl.lit(row['gene']).alias('gene'))

        methylation_data = methylation_data.filter(pl.col('position').is_between(row['start']-buffer,row['end']+buffer))
        all_methylation_data.append(methylation_data)
    return pl.concat(all_methylation_data).drop('gene')


        
if __name__ =='__main__':
    sample_id = '244'
    target_gene_df = load_target_gene_data()
    
    
    methylation_data = load_methylation_data(target_gene_df).unique()
    
    window_buffer = 0
    #gene_data = get_gene_data('hg38')


    plt_index = 0
    for row in target_gene_df.iter_rows(named=True):
        print(row)
        fire_data = plot_helper.load_fire_calls(row["sample_id"],row['chromosome'])
        fire_regions = plot_helper.get_fire_data_with_region(fire_data,row,window_buffer=window_buffer)
        out_dir = Path(f'/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/read_level_data')
        out_dir.mkdir(exist_ok=True,parents=True)
        out_path = out_dir/f'{row["gene"]}.pdf'
        
        plot_region(methylation_data,['tumor_read','haplotag'],row['chromosome'],row['start'],row['end'],out_path,fire_regions=fire_regions,window_buffer=window_buffer)
        print(out_path)
        plt_index+=1
        