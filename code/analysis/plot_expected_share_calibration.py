import sys
import plotting_tools
import polars as pl
import numpy as np

from pathlib import Path


import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0,'../processing')
from cn_loader import load_cn

#claude 4.1a
def format_pvalue(p, precision=2):
    """Format p-value as 'y × 10^z' instead of scientific notation."""
    if p == 0:
        return "0"
    elif p < 0.001:  # Only use scientific notation for very small p-values
        exponent = int(np.floor(np.log10(abs(p))))
        mantissa = p / 10**exponent
        return f"{mantissa:.{precision}f} × $10^{{{exponent}}}$"
    else:
        return f"{p:.{precision}f}" 

def load_read_data(path):
    
    read_data = pl.scan_parquet(path).select(['read_index','tumor_read'])

    read_data =read_data.with_columns(pl.col('tumor_read').cast(pl.UInt8))
    return read_data

def load_read_labels(sample_id):
    in_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/labelled_data')
    in_path = in_dir/f'{sample_id}_TU_labelled_data.parquet'
    return load_read_data(in_path)
def load_read_extent(sample_id):
    read_extent_dir = Path('/hot/user/tobybaker/CellTypeClassifier/data/read_extent/')
    read_extent_path = read_extent_dir/f'{sample_id}_TU_read_extent.parquet'
    read_extent = pl.scan_parquet(read_extent_path).with_columns(pl.col('chromosome').cast(pl.Categorical))
    return read_extent


def load_predictions(sample_id:str,add_normal:bool):
    read_extent = load_read_extent(sample_id)
    read_labels = load_read_labels(sample_id).select('read_index')

    
    
    
    base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/main_predictions/')
    predictions_dir = base_dir/f'{sample_id}_TU_add_normal_{add_normal}/full_datasets'
    
    predictions_path =  predictions_dir /f'train_{sample_id}_TU_add_normal_{add_normal}_out_{sample_id}_TU_all_reads.parquet'
    
    predictions = pl.scan_parquet(predictions_path)
    
    #predictions = predictions.rename({'Sample_ID':'sample_id','Read_Index':'read_index','Chromosome':'chromosome','Tumor_Probability':'tumor_probability'})
    predictions = predictions.with_columns(pl.col('chromosome').cast(pl.Categorical))
    predictions = predictions.join(read_extent,how='inner',on=['chromosome','read_index'])
    predictions = predictions.join(read_labels,on=['read_index'],how='anti')
    predictions = predictions.with_columns(pl.col('tumor_probability')>=0.5)
    
    return predictions


def get_share_store(sample_id:str,add_normal:bool):
    cn = load_cn(f'{sample_id}_TU')
    

    predictions = load_predictions(sample_id,add_normal=add_normal).collect()

    run_data = []
    for chromosome,cn_chromosome in cn.group_by('chromosome'):
        
        read_chromosome = predictions.filter(pl.col('chromosome')==chromosome[0])
    
        for cn_row in cn_chromosome.iter_rows(named=True):
            
            segment_id = f"{cn_row['chromosome']}-{cn_row['segment_end']}-{cn_row['segment_start']}"
            
            matching_reads = read_chromosome.filter((pl.col('reference_start')<=cn_row['segment_end']) &(cn_row['segment_start']<=pl.col('reference_end')))
            n_matching_reads = matching_reads.height
            if n_matching_reads ==0:
                continue
            avg_prob = np.nanmean(matching_reads['tumor_probability'])
            std = np.nanstd(matching_reads['tumor_probability'])
            expected_share = cn_row['total_cn']*cn_row['purity']/(cn_row['total_cn']*cn_row['purity']+ cn_row['normal_total_cn']*(1-cn_row['purity']))
            
            
            segment_length = cn_row['segment_end']-cn_row['segment_start']

            segment_data = {'chromosome':chromosome,'segment_id':segment_id,'average_probability':avg_prob,'expected_share':expected_share,'std':std,'segment_length':segment_length,'n_reads':n_matching_reads,'total_cn':cn_row['total_cn'],'purity':cn_row['purity'],'normal_total_cn':cn_row['normal_total_cn']}
            run_data.append(segment_data)
    run_data = pl.DataFrame(run_data).with_columns(pl.lit(sample_id).alias('sample_id'))
    

    return run_data

def load_all_share_data(sample_ids,add_normal):
    all_share_data = []
    for sample_id in sample_ids:
        share_store = get_share_store(sample_id,add_normal)
        all_share_data.append(share_store)
        print('loaded',sample_id)
        
    return pl.concat(all_share_data)

def summarise_share_data(share_data,sample_ids,min_segment_length=1e6,min_n_reads=1000,min_segments=5):
    share_data = share_data.filter(pl.col('segment_length')>=min_segment_length)
    share_data = share_data.filter(pl.col('n_reads')>=min_n_reads)
    share_data_summary = (
    share_data
    .group_by("sample_id", "total_cn", "normal_total_cn", "expected_share")
    .agg(
        pl.col("average_probability").mean().alias("mean"),
        pl.col("average_probability").std().alias("std"),
        pl.col("average_probability").count().alias("size"),
    )
    )
    share_data_summary = share_data_summary.filter(pl.col('size')>=min_segments)
    share_data_summary = share_data_summary.with_columns(pl.col('sample_id').cast(pl.Enum(sample_ids)))
    
    return share_data_summary.sort('sample_id')


def plot_share_data_summary(share_data_summary,add_normal:bool):
    plot_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/model_performance')
    fig,ax = plt.subplots(1,1,figsize=(6,3.7))
    #sample_mapping = {'BS14772':'Prostate A','BS15145':'Prostate B','216':'Ovarian A','244':'Ovarian B','264':'Ovarian C'}
    for sample_id,sample_data in share_data_summary.group_by('sample_id',maintain_order=True):
        sample_id = sample_id[0]
        ax.errorbar(sample_data['expected_share'],sample_data['mean'], yerr=sample_data['std'], fmt='none', capsize=3, capthick=1, color='black',alpha=0.2)
        ax.scatter(sample_data['expected_share'],sample_data['mean'],label=plotting_tools.sample_mapping[sample_id],s=50,alpha=0.8,c=plotting_tools.get_sample_color_scheme()[sample_id])
        
    
    r,p = pearsonr(share_data_summary['expected_share'],share_data_summary['mean'])
    ax.plot([0,1],[0,1],linestyle='dashed',color='red',lw=2,label='Perfect Correlation')
    ax.legend(ncol=2)
    ax.set_xlabel('Expected proportion of tumor reads')
    ax.set_ylabel('Predicted proportion of tumor reads')
    ax.set_title(f"Pearson's : {r:.3f} P = {format_pvalue(p)}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir/f'expected_share_correlation_{add_normal}.png')
    plt.savefig(plot_dir/f'expected_share_correlation_{add_normal}.pdf')
if __name__ =='__main__':
    sample_ids = ['BS14772','BS15145','216','244','264','053']
    for add_normal in [True,False]:
        share_data = load_all_share_data(sample_ids,add_normal)
        share_data_summary = summarise_share_data(share_data,sample_ids)
        
        plot_share_data_summary(share_data_summary,add_normal)