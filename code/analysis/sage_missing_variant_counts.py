import polars as pl
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0,'../processing')
from variant_loader import load_long_read_variants
from supported_vs_unsupported_variant_distribution import load_all_variant_data
import cn_loader
import plotting_tools

from scipy.stats import spearmanr,pearsonr

NON_TRAIN_CHROMOSOMES = ['chr4','chr5','chr21','chr22']
def load_read_table(sample_id:str):
    in_dir = Path('/hot/user/tobybaker/ROCIT_Paper/output/read_variant_store/variant_long_read_bam_short_read')
    filename = f'{sample_id}_NL_reads.parquet'
    in_path = in_dir/filename
    in_df = pl.read_parquet(in_path)
    
    
    in_df = in_df.group_by(['chromosome','position','ref','alt','SAGE_filter_status']).agg(
        pl.col('contains_snv').sum().alias("normal_alt_count"),
        pl.col('contains_snv').len().alias("normal_total_count"),
    )
    
    return in_df
    
def load_variant_counts(sample_id:str):
    read_table = load_read_table(sample_id)
    
    variant_table = load_long_read_variants(f'{sample_id}_TU',pass_filter=True)
    variant_table = variant_table.join(read_table,how='left',on=['chromosome','position','ref','alt'],coalesce=True)
    variant_table= variant_table.with_columns(
    pl.col("normal_alt_count", "normal_total_count").fill_null(0),
    pl.col("SAGE_filter_status").fill_null('missing'),
    pl.lit(plotting_tools.get_sample_mapping()[sample_id]).alias('sample_id'),
    )

    variant_table = variant_table.with_columns((pl.col('SAGE_filter_status')=='pass').alias('SAGE_supported'))

    return variant_table

def load_combined_variant_counts(sample_ids):
    store = []
    for sample_id in sample_ids:
        variant_counts = load_variant_counts(sample_id)
        store.append(variant_counts)
    return pl.concat(store)

def load_variant_prediction_status():
    all_variant_data = load_all_variant_data()
    all_variant_data = all_variant_data.collect()
    all_variant_data = all_variant_data.filter(~pl.col('in_sage'))

    all_variant_data = all_variant_data.group_by(['sample_id']).agg((pl.col('tumor_probability')<0.5).mean().alias('prop_prob_low'))
    return all_variant_data
def get_expected_share_df(sample_ids):
    store = []
    for sample_id in sample_ids:
        cn = cn_loader.load_cn(f'{sample_id}_TU')
        cn = cn.filter(pl.col('chromosome').is_in(NON_TRAIN_CHROMOSOMES))
        cn = cn.with_columns((pl.col('total_cn')*pl.col('purity')/(pl.col('total_cn')*pl.col('purity') + pl.col('normal_total_cn')*(1.0-pl.col('purity')))).alias('expected_share'))
        expected_share = np.average(cn['expected_share'],weights=cn['segment_length'])
        store.append({'sample_id':plotting_tools.get_sample_mapping()[sample_id],'expected_share':expected_share})
    return pl.DataFrame(store)
def plot_missing_expected_share(plot_table):

    fig,ax = plt.subplots(1,1,figsize=(6,4))
    #sample_mapping = {'BS14772':'Prostate A','BS15145':'Prostate B','216':'Ovarian A','244':'Ovarian B','264':'Ovarian C'}
    for sample_id,sample_data in plot_table.group_by('sample_id',maintain_order=True):
        sample_id = sample_id[0]
        ax.scatter(1.0-sample_data['expected_share'],sample_data['prop_prob_low'],label=sample_id,s=50,alpha=0.8,c=plotting_tools.get_sample_color_scheme()[sample_id])
        
    ax.legend()
    ax.set_xlabel('Expected proportion of non-tumor reads')
    ax.set_ylabel('Proportion of variant reads\npredicted non-tumor')
    ax.set_title(f"Reads containing SAGE unsupported variants")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plot_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/snv_calling')
    plt.savefig(plot_dir/f'expected_share_prop_non_tumor.png')
    plt.savefig(plot_dir/f'expected_share_prop_non_tumor.pdf')
if __name__ =='__main__':
    sample_ids = ['216','244','264','053','BS14772','BS15145']
    variant_prediction_status = load_variant_prediction_status()
    expected_share_df  =get_expected_share_df(sample_ids)

    combined_variant_count = load_combined_variant_counts(sample_ids)
    combined_variant_count = combined_variant_count.filter(~pl.col('SAGE_supported'))
    combined_variant_count = combined_variant_count.filter(pl.col('chromosome').is_in(NON_TRAIN_CHROMOSOMES))
    

    alt_matched_normal= combined_variant_count.group_by('sample_id').agg((pl.col('normal_alt_count')>=3).mean().alias('proportion_alt_matched_normal'))
    plot_table = variant_prediction_status.join(alt_matched_normal,how='inner',on='sample_id')
  
    plot_table = plot_table.join(expected_share_df,how='inner',on='sample_id')
    text_out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/text')
    plot_table.write_csv(text_out_dir/'proportion_variant_reads.tsv',separator='\t')
    exit()
    plot_missing_expected_share(plot_table)
    