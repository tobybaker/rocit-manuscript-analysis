import os
import polars as pl
import numpy as np
#import plotting_tools
import matplotlib.pyplot as plt

from pathlib import Path

from rocit.constants import HUMAN_CHROMOSOME_ENUM

import sys

sys.path.insert(0,'../analysis')
import plotting_tools
def load_tumor_predictions(sample_id:str):
    base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/main_predictions/')
    sample_dir = base_dir/f'{sample_id}_TU_add_normal_False/full_datasets'
    filename = f'train_{sample_id}_TU_add_normal_False_out_{sample_id}_TU_all_reads.parquet'
    in_path = sample_dir/filename
    in_df = pl.scan_parquet(in_path)
    in_df = in_df.with_columns(pl.col('sample_id').cast(pl.Categorical),pl.col('read_index').cast(pl.Categorical),pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM))
    return in_df


def load_variant_data(sample_id:str):
    
    in_dir = Path('/hot/user/tobybaker/ROCIT_Paper/output/long_read_variants_with_short_read_status/')
    filename = f'{sample_id}_TU_reads.parquet'
    filepath = in_dir/filename
    in_df = pl.scan_parquet(filepath)
    predictions = load_tumor_predictions(sample_id)

    
    in_df = in_df.join(predictions,how='inner',on=['chromosome','read_index'],coalesce=True)
    
    in_df = (
        in_df
        .with_columns(
            (pl.col("tumor_probability") > 0.5).alias("tumor_read"),
            (pl.col("tumor_alt_count") / (pl.col("tumor_alt_count") + pl.col("tumor_ref_count"))).alias("vaf"),
        )
        .filter(
            pl.col("contains_snv"),
            pl.col("chromosome").is_in(["chr4", "chr5", "chr21", "chr22"]),
        )
        .with_columns(
            (pl.col("vaf").rank() / pl.col("vaf").count()).alias("vaf_rank"),
            (pl.col('SAGE_filter_status')=='pass').alias('in_sage')
        )
    )
    #in_df['In_SAGE'] = in_df['SAGE_Status'] =='PASS'


    return in_df

def load_all_variant_data():
    all_variant_data = []
    sample_ids = ['216','244','264','053','BS14772','BS15145']
    for sample_id in sample_ids:
        
        sample_variant_data = load_variant_data(sample_id)
        sample_variant_data=sample_variant_data.with_columns(pl.lit(plotting_tools.get_sample_mapping()[sample_id]).alias('sample_id'))
        all_variant_data.append(sample_variant_data)
    return pl.concat(all_variant_data).sort('sample_id')


def plot_variant_histogram(in_df,ax,title,short_legend=False):
    bins = np.linspace(0,1,51)

    color_mapping = {True:'#11b4f5',False:'#500082'}
    for sage_status,sage_table in in_df.group_by('in_sage'):
        sage_status = sage_status[0]
        if short_legend:
            sage_label = f'(n={sage_table.height:,})'
        else:
            sage_label = f'SAGE supported variant' if sage_status else 'SAGE unsupported variant'
            sage_label = f'{sage_label} (n={sage_table.height:,})'
        ax.hist(sage_table['tumor_probability'],bins=bins,density=True,label=sage_label,alpha=0.5,color=color_mapping[sage_status])
    ax.legend()
    ax.set_xlabel('Tumor Read Probability')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_sage_filter_status(all_variant_data):
    variant_data = all_variant_data[['Sample_ID','ID','SAGE_Status']].drop_duplicates()
    
    n_sage_pass = (variant_data['SAGE_Status']=='PASS').sum()
    n_sage_missing = (variant_data['SAGE_Status']=='Missing').sum()
    n_sage_filtered = len(variant_data) - n_sage_pass-n_sage_missing
    fig,ax = plt.subplots(1,1,figsize=(4.5,4))

    ax.bar([0,1,2],[n_sage_pass,n_sage_missing,n_sage_filtered],color='#0453B4')
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['SAGE Pass','SAGE Uncalled','SAGE Filtered'])
    ax.set_ylabel('Number of DeepSomatic mutations')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('sage_filter_status.png')
    plt.savefig('sage_filter_status.pdf')

def load_variant_labels():
    in_dir = '/hot/user/selinawu/project-ROCIT/sSNV/analyze_variant_calls/analysis_outputs/'
    df_store = []
    for filename in ['sageNP_dsPASS_per_snv_reasoning.tsv','sageNP_dsPASS_per_indel_reasoning.tsv']:
        filepath = os.path.join(in_dir,filename)
        in_df = pd.read_csv(filepath,sep="\t")
        
        in_df = in_df.rename(columns={'chrom':'Chromosome','pos':'Position','reason.category':'Reason','sample':'Sample_ID'})
        in_df = in_df[['Sample_ID','Chromosome','Position','Reason']]
        in_df['Sample_ID'] = in_df['Sample_ID'].map(plotting_tools.get_sample_mapping())
        df_store.append(in_df.copy())
    return pd.concat(df_store)

def plot_reason_histogram(all_variant_data):
    reason_data = all_variant_data[(all_variant_data['SAGE_Status']!='PASS') & (~all_variant_data['Reason'].isnull())].copy()
    bins = np.linspace(0,1,51)
    fig,axs = plt.subplots(3,2,figsize=(10,10))
    axs = axs.flatten()
    plt_count = 0
    for reason,reason_table in reason_data.groupby('Reason'):
        axs[plt_count].hist(reason_table['Probability'],bins=bins,color='#500082')
        reason_string = ' '.join([x.capitalize() for x in reason.split('.')])
        axs[plt_count].set_title(f'{reason_string} (n={len(reason_table)})')
        axs[plt_count].set_xlabel('Tumor Read Probability')
        axs[plt_count].set_ylabel('Number of reads')
        plt_count +=1
    fig.suptitle('SAGE Unsupported Variants')
    plt.tight_layout()
    plt.savefig('read_probability_by_sage_filter.png')
    plt.savefig('read_probability_by_sage_filter.pdf')
    
if __name__ =='__main__':

    

    all_variant_data = load_all_variant_data()
    all_variant_data = all_variant_data.collect()
    all_variant_data.write_parquet('/scratch/all_variant_data.parquet')
    all_variant_data = pl.read_parquet('/scratch/all_variant_data.parquet')


    fig,ax = plt.subplots(1,1,figsize=(6,4))

    
    plot_variant_histogram(all_variant_data,ax,'Reads containing somatic variants')
    
    plt.savefig('somatic_variant_reads_tumor_probability.png')
    plt.savefig('somatic_variant_reads_tumor_probability.pdf')
    

    fig,axs = plt.subplots(2,4,figsize=(10,5))
    axs = axs.flatten()
    plt_count = 0
    for sample_id,sample_data in all_variant_data.group_by('sample_id',maintain_order=True):
        print(sample_id)
        if sample_id[0] =='053_TU':
            sample_data.to_pandas()[['chromosome','position','ref','alt','tumor_ref_count','tumor_alt_count','SAGE_filter_status']].drop_duplicates().sort_values(by=['SAGE_filter_status','tumor_ref_count']).to_csv('test.tsv',sep="\t",index=False)
        print('==========')
        plot_variant_histogram(sample_data,axs[plt_count],title=sample_id[0],short_legend=True)
        plt_count+=1
    fig.suptitle('Reads containing somatic variants')
    axs[-1].axis('off')  # Hide axes frame, ticks, etc.
    axs[-2].set_visible(False)

    # Making the bigger text legend on the final axis
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, ['SAGE unsupported variant','SAGE supported variant'], loc='center left')

    plt.tight_layout()
    plt.savefig('somatic_variant_reads_tumor_probability_by_sample.png')
    plt.savefig('somatic_variant_reads_tumor_probability_by_sample.pdf')


    
    