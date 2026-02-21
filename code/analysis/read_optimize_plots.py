import polars as pl
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import plotting_tools
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

from pathlib import Path
from datetime import datetime
from numba import njit,prange

from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from rocit.constants import HUMAN_CHROMOSOME_ENUM

MODIFICATION_THRESHOLD = 0.1
PROBABILITY_THRESHOLD = 0.2

MAIN_DIR = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/')
PLOT_DIR = MAIN_DIR /'plots/read_interpretation'
SUPPLEMENTARY_PLOT_DIR = MAIN_DIR/'plots/supplementary_figures'
LOG_PATH = MAIN_DIR /'text/read_optimize_out.txt'
MAIN_PENALTY=15

TYPE_COLOR_SCHEME = ['#9bcf61','#D3737B']
def get_significance_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'
def write_to_log(text:str,append:bool=True):
    mode = 'a' if append else 'w'
    with open(LOG_PATH,mode) as out:
        out.write(f'{text}\n')


def get_sequential_colors(x,min_val=0.05,max_val=0.95):
    
    cmap = plt.cm.viridis
    colors = [cmap((i / (x - 1)*(max_val-min_val))+min_val) if x > 1 else cmap(0.5) for i in range(x)]
    return colors

import polars as pl
import numpy as np
import re


def bin_probs(prob_series: pl.Series) -> pl.Series:
    """
    Bin a probability Series into 10 fixed-width bins.
    Returns a Series of bin labels (strings representing bin ranges).
    """
    bins = np.linspace(0, 1, 11).tolist()
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]
    return prob_series.cut(breaks=bins[1:-1], labels=labels)


def load_dataframe(filepath: str) -> pl.DataFrame:
    # Load parquet
    in_df = pl.read_parquet(filepath)
    
    in_df = in_df.filter(
        (pl.col("original_probability") < PROBABILITY_THRESHOLD)
        | (pl.col("original_probability") > 1 - PROBABILITY_THRESHOLD)
    )


    sample_id = filepath.split("/")[-2]
    penalty = int(float(re.search(r"penalty_(\d+\.\d+)", filepath).group(1)))

    # Add tumor prediction column
    in_df = in_df.with_columns(
        (pl.col("original_probability") > 0.5).alias("tumor_predicted_read")
    )

    # Add metadata columns
    in_df = in_df.with_columns(pl.lit(penalty).alias("penalty"))
    in_df = in_df.with_columns(pl.lit(sample_id).alias("sample_id"))

    # Add binned methylation columns
    in_df = in_df.with_columns(
        bin_probs(in_df["original_methylation"]).alias("original_bin")
    )
    in_df = in_df.with_columns(
        bin_probs(in_df["modified_methylation"]).alias("modified_bin")
    )

    # Identify CpGs with significant methylation changes
    in_df = in_df.with_columns(
        (
            (pl.col("original_methylation") - pl.col("modified_methylation")).abs()
            > MODIFICATION_THRESHOLD
        ).alias("switched_cpg")
    )

    # Rename position column
    in_df = in_df.rename({"positions": "position"})

    # Identify reads that switched classification
    in_df = in_df.with_columns(
        (
            pl.col("tumor_predicted_read")
            & (pl.col("modified_probability") < PROBABILITY_THRESHOLD)
        ).alias("tumor_to_non_tumor")
    )

    in_df = in_df.with_columns(
        (
            ~pl.col("tumor_predicted_read")
            & (pl.col("modified_probability") > (1 - PROBABILITY_THRESHOLD))
        ).alias("non_tumor_to_tumor")
    )

    # Mark successful modifications
    in_df = in_df.with_columns(
        (pl.col("tumor_to_non_tumor") | pl.col("non_tumor_to_tumor")).alias("successful")
    )
    in_df = in_df.with_columns(pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM))
    return in_df

def get_read_data(sample_data: pl.DataFrame) -> pl.DataFrame:
    group_cols = [
        "read_index",
        "chromosome",
        "sample_id",
        "penalty",
        "tumor_predicted_read",
        "original_probability",
        "modified_probability",
        "successful",
    ]

    # Aggregate switched CpG counts per read
    read_data = sample_data.group_by(group_cols).agg(
        pl.col("switched_cpg").sum().alias("n_switch"),
        pl.col("switched_cpg").count().alias("n_cpgs"),
    )

    # Calculate fraction of CpGs switched
    read_data = read_data.with_columns(
        (pl.col("n_switch") / pl.col("n_cpgs")).alias("frac_switch")
    )

    return read_data
def load_sample_data(sample_ids):
    in_dir = '/hot/user/tobybaker/ROCIT_Paper/read_optimizations'
    
    sample_data = []
    for sample_id in sample_ids:
        sample_dir =f'{in_dir}/{sample_id}'
        
        for filename in os.listdir(sample_dir):
            if not filename.endswith('.parquet'):
                continue
            
            filepath = os.path.join(sample_dir,filename)
            in_df = load_dataframe(filepath)
            sample_data.append(in_df)
        
    return pl.concat(sample_data)



def get_sample_success_proportions(read_data):

    read_counts = read_data.group_by(['sample_id','tumor_predicted_read','penalty']).agg([
        pl.col('successful').len().alias('n_observations'),
        pl.col('successful').sum().alias('n_success')
    ])
    read_counts = read_counts.with_columns((pl.col('n_success')/pl.col('n_observations')).alias('proportion'))

    ci_low, ci_high = proportion_confint(
    count=read_counts['n_success'],
    nobs=read_counts['n_observations'],
    alpha=0.05,  # for 95% CI (1 - 0.05 = 0.95)
    method='wilson'
)   
    read_counts = read_counts.with_columns(pl.lit(ci_low).alias('ci_low'),pl.lit(ci_high).alias('ci_high'))
    return read_counts
def plot_sample_success_proportions(read_data):

    color_scheme = {True:'blue',False:'red'}
    legend_mapping = {True:'Tumor to Non-Tumor',False:"Non-Tumor to Tumor"}
    read_counts = get_sample_success_proportions(read_data)
    
    fig,axs = plt.subplots(2,3,figsize=(10,5))
    axs = axs.flatten()
    sample_count = 0
    
    bar_width = 0.3
    for sample_id,sample_counts in read_counts.partition_by('sample_id', as_dict=True).items():
        sample_id = sample_id[0]
        ax = axs[sample_count]
        tumor_offset = -bar_width
        for tumor_predicted_read,tumor_table in sample_counts.partition_by('tumor_predicted_read', as_dict=True).items():
            tumor_predicted_read = tumor_predicted_read[0]
            yerr = (tumor_table['proportion']-tumor_table['ci_low'],tumor_table['ci_high']-tumor_table['proportion'])
            ax.bar(np.arange(len(tumor_table))+tumor_offset,tumor_table['proportion'],color=color_scheme[tumor_predicted_read],yerr=yerr,capsize=5,label=legend_mapping[tumor_predicted_read],align='edge',width=bar_width)
            tumor_offset += bar_width
        ax.set_xticks(np.arange(read_data['penalty'].n_unique()))
        ax.set_xticklabels(sorted(read_data['penalty'].unique()))
        ax.set_ylabel('Proportion of reads converted')
        ax.set_xlabel('CpG Modification\npenalty')
        ax.legend(title='Original Predicition')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(plotting_tools.get_sample_mapping()[sample_id.split('_')[0]])

        sample_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'sample_success_reads.png')
    plt.savefig(PLOT_DIR / 'sample_success_reads.pdf')

def plot_frac_success_by_type(sample_data):
    sample_data = sample_data.filter(pl.col('penalty')==MAIN_PENALTY)
    sample_data =sample_data.select(['penalty','sample_id','read_index','tumor_predicted_read','successful']).unique()
    
    switch_by_type = sample_data.group_by("sample_id", "tumor_predicted_read").agg(pl.col("successful").mean())

    switch_by_type_agg = switch_by_type.group_by("tumor_predicted_read")
    switch_by_type_agg = switch_by_type_agg.agg(
        pl.col("successful").mean().alias("mean"),
        pl.col("successful").std().alias("std")
    )
    switch_by_type_agg = switch_by_type_agg.sort("tumor_predicted_read", descending=True)
        
    fig,ax = plt.subplots(1,1,figsize=(2.8,3))

  
    ax.bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
    ax.set_xticks(np.arange(len(switch_by_type_agg)))

    n_tumor_to_non_tumor = sample_data['tumor_predicted_read'].sum()
    non_tumor_to_n_tumor = len(sample_data)-n_tumor_to_non_tumor

    #ax.set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
    ax.set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
    ax.set_ylabel('Proportion of reads converted')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_success_by_type.png')
    plt.savefig(PLOT_DIR / 'frac_success_by_type.pdf')

def plot_frac_success_by_type_penalty(sample_data):
    
    sample_data =sample_data.select(['penalty','sample_id','read_index','tumor_predicted_read','successful']).unique().sort('penalty')
    fig,ax = plt.subplots(1,3,figsize=(8,3))
    plt_count = 0
    write_to_log('====FRAC SUCCESS BY PENALTY====')
    for penalty,penalty_data in sample_data.partition_by('penalty', as_dict=True).items():
        penalty = penalty[0]
        switch_by_type = penalty_data.group_by("sample_id", "tumor_predicted_read").agg(pl.col("successful").mean())

        switch_by_type_agg = switch_by_type.group_by("tumor_predicted_read")
        switch_by_type_agg = switch_by_type_agg.agg(
            pl.col("successful").mean().alias("mean"),
            pl.col("successful").std().alias("std")
        )
        switch_by_type_agg = switch_by_type_agg.sort("tumor_predicted_read", descending=True)

        write_to_log(f'penalty {penalty}')
        write_to_log(switch_by_type_agg.to_pandas().to_string())
    
        ax[plt_count].bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
        ax[plt_count].set_xticks(np.arange(len(switch_by_type_agg)))
        n_tumor_to_non_tumor = penalty_data['tumor_predicted_read'].sum()
        non_tumor_to_n_tumor = len(penalty_data)-n_tumor_to_non_tumor

        #ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
        ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
        ax[plt_count].set_ylabel('Proportion of reads converted')
        ax[plt_count].spines['top'].set_visible(False)
        ax[plt_count].spines['right'].set_visible(False)
        ax[plt_count].set_ylim(0,1.05)
        ax[plt_count].set_title(f'Sparsity penalty\n{penalty}')

        plt_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_success_by_type_penalty.png')
    plt.savefig(PLOT_DIR / 'frac_success_by_type_penalty.pdf')

def plot_frac_switch_by_type_penalty(sample_data):
    sample_data = sample_data.filter(pl.col('successful')).sort('penalty')
    fig,ax = plt.subplots(1,3,figsize=(8,3))
    plt_count = 0

    write_to_log('====FRAC SWITCH BY PENALTY===')
    for penalty,penalty_data in sample_data.partition_by('penalty', as_dict=True).items():
        switch_by_type = penalty_data.group_by(['sample_id', 'tumor_predicted_read']).agg(
            pl.col('switched_cpg').mean()
        )

        switch_by_type_agg = switch_by_type.group_by('tumor_predicted_read').agg([
            pl.col('switched_cpg').mean().alias('mean'),
            pl.col('switched_cpg').std().alias('std')
        ])

        switch_by_type_agg = switch_by_type_agg.sort('tumor_predicted_read', descending=True)
        
        write_to_log(f'penalty {penalty}')
        write_to_log(switch_by_type_agg.to_pandas().to_string())
    
        ax[plt_count].bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
        ax[plt_count].set_xticks(np.arange(len(switch_by_type_agg)))

        ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
        ax[plt_count].set_ylabel('Proportion of CpGs Perturbed')
        ax[plt_count].spines['top'].set_visible(False)
        ax[plt_count].spines['right'].set_visible(False)
        ax[plt_count].set_ylim(0,0.15)
        ax[plt_count].set_title(f'Sparsity penalty\n{penalty[0]}')

        plt_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_switch_by_type_penalty.png')
    plt.savefig(PLOT_DIR / 'frac_switch_by_type_penalty.pdf')

def plot_frac_switch_by_type(sample_data):

    sample_data = sample_data.filter(
        (pl.col('penalty') == MAIN_PENALTY) & 
        (pl.col('successful'))
    )
    switch_by_type = sample_data.group_by(['sample_id', 'tumor_predicted_read']).agg(
        pl.col('switched_cpg').mean()
    )

    switch_by_type_agg = switch_by_type.group_by('tumor_predicted_read').agg([
        pl.col('switched_cpg').mean().alias('mean'),
        pl.col('switched_cpg').std().alias('std')
    ])

    switch_by_type_agg = switch_by_type_agg.sort('tumor_predicted_read', descending=True)
    
    fig,ax = plt.subplots(1,1,figsize=(2.8,3))

  
    ax.bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
    ax.set_xticks(np.arange(len(switch_by_type_agg)))

    ax.set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
    ax.set_ylabel('Proportion of CpGs Perturbed')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_switch_by_type.png')
    plt.savefig(PLOT_DIR / 'frac_switch_by_type.pdf')

def plot_switch_directions_by_type(sample_data):
    sample_data = sample_data.filter(
        (pl.col('switched_cpg')) & 
        (pl.col('successful')) &
        (pl.col('penalty')==MAIN_PENALTY)
    )
    sample_data = sample_data.with_columns((pl.col('original_methylation') < pl.col('modified_methylation')).alias('to_hyper'))

    switch_by_type = sample_data.group_by("sample_id", "tumor_predicted_read").agg(pl.col("to_hyper").mean())

    switch_by_type_agg = (
        switch_by_type
        .group_by("tumor_predicted_read")
        .agg(
            pl.col("to_hyper").mean().alias("mean"),
            pl.col("to_hyper").std().alias("std"),
        )
        .sort("tumor_predicted_read", descending=True)
    )

    
    fig,ax = plt.subplots(1,1,figsize=(2.8,3))

  
    ax.bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
    ax.set_xticks(np.arange(len(switch_by_type_agg)))
    n_tumor_to_non_tumor = sample_data['tumor_predicted_read'].sum()
    non_tumor_to_n_tumor = len(sample_data)-n_tumor_to_non_tumor

    #ax.set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
    ax.set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
    ax.set_ylabel('Proportion of perturbations\nincreasing methylation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'switch_directions_by_type.png')
    plt.savefig(PLOT_DIR / 'switch_directions_by_type.pdf')

def plot_switch_directions_by_type_penalty(sample_data):
    sample_data = sample_data.filter(
        (pl.col('switched_cpg')) & 
        (pl.col('successful'))
    )
    sample_data = sample_data.with_columns((pl.col('original_methylation') < pl.col('modified_methylation')).alias('to_hyper'))
    sample_data = sample_data.sort('penalty')
    fig,ax = plt.subplots(1,3,figsize=(8,3))
    plt_count = 0
    write_to_log('===Switch_Directions penalty===')
    for penalty,penalty_data in sample_data.partition_by('penalty', as_dict=True).items():
        switch_by_type = penalty_data.group_by("sample_id", "tumor_predicted_read").agg(pl.col("to_hyper").mean())

        switch_by_type_agg = (
            switch_by_type
            .group_by("tumor_predicted_read")
            .agg(
                pl.col("to_hyper").mean().alias("mean"),
                pl.col("to_hyper").std().alias("std"),
            )
            .sort("tumor_predicted_read", descending=True)
        )
        write_to_log(f'penalty {penalty}')
        write_to_log(switch_by_type_agg.to_pandas().to_string())

        
        ax[plt_count].bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
        ax[plt_count].set_xticks(np.arange(len(switch_by_type_agg)))
        n_tumor_to_non_tumor = penalty_data['tumor_predicted_read'].sum()
        non_tumor_to_n_tumor = len(penalty_data)-n_tumor_to_non_tumor
        ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
        #ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
        ax[plt_count].set_ylabel('Proportion of perturbations\nadding methylation')
        ax[plt_count].spines['top'].set_visible(False)
        ax[plt_count].spines['right'].set_visible(False)
        ax[plt_count].set_ylim(0,1.05)
        ax[plt_count].set_title(f'Sparsity penalty\n{penalty[0]}')

        plt_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'switch_directions_by_type_penalty.png')
    plt.savefig(PLOT_DIR / 'switch_directions_by_type_penalty.pdf')


def load_cell_map_data():
    base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/')
    cell_type_path = base_dir/'cell_type_average_methylation_atlas.parquet'
    cell_map_df = pl.scan_parquet(cell_type_path).with_columns(pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM))
    return cell_map_df

def load_sample_dist_df(sample_id):
    base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/cpg_methylation_distribution')

    sample_dist_path = base_dir/f'{sample_id}_methylation_distribution.parquet'
    sample_dist_df = pl.scan_parquet(sample_dist_path).select(['chromosome','position','methylation_percentile_50'])
    
    sample_dist_df = sample_dist_df.with_columns(pl.lit(sample_id).alias("sample_id"),pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM))

    return sample_dist_df
        
def get_sample_distribution_data(sample_data):
    sample_ids = sample_data['sample_id'].unique().to_list()
    sample_distribution_store = []
    for sample_id in sample_ids:
        sample_dist = load_sample_dist_df(sample_id)
        sample_distribution_store.append(sample_dist)
    return pl.concat(sample_distribution_store)
def get_supplementary_annotations(sample_data):
    cell_map_data = load_cell_map_data()
    sample_distribution = get_sample_distribution_data(sample_data)
    
    sample_data = sample_data.lazy().join(cell_map_data,on=['chromosome','position'],how='inner')
    sample_data = sample_data.join(sample_distribution,on=['sample_id','chromosome','position'],how='inner')
    return sample_data.collect()
def plot_supplementary_variance_violin(sample_data):
    bool_order =(False,True)
    
    
    sample_data = (
    sample_data
    .filter(
        pl.col("successful"),
        pl.col("penalty") == MAIN_PENALTY,
    )
    )
    sample_data = get_supplementary_annotations(sample_data)
    cell_type_cols = [col for col in sample_data.columns if col.startswith('average_methylation_')]
    
    median_data = sample_data.select(['switched_cpg','methylation_percentile_50'])

    median_data = median_data.drop_nulls()
    median_plot = [median_data.filter(pl.col('switched_cpg')==x)['methylation_percentile_50'].to_numpy() for x in bool_order]
    
    cell_type_data = sample_data.with_columns(
        pl.concat_list(cell_type_cols)
      .list.eval(pl.element().drop_nulls().drop_nans().std())
      .list.first()
      .alias("std")
    
    )
    cell_type_data = cell_type_data.select(['switched_cpg','std'])


    cell_type_data = cell_type_data.drop_nulls()
    
    cell_type_plot = [cell_type_data.filter(pl.col('switched_cpg')==x)['std'].to_numpy() for x in [False, True]]

    write_to_log('==== Cell Type  & Median Data')
    for data_label,dataset in [('cell_type_STD',cell_type_plot),('Median',median_plot)]:
        for i,bool_val in enumerate(bool_order):
            log_text = f'{data_label} - Perturbed CpG {bool_val} - Median = {np.median(dataset[i]):.4f}'
            write_to_log(log_text)
    
    tick_labels = ['Non-Perturbed\nCpG', 'Perturbed\nCpG']
    
    fig_0, ax_0 = plt.subplots(1, 1, figsize=(2.5, 3))
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(2.5, 3))
    axs = [ax_0,ax_1]
    color_scheme = ['#9e9e9e', '#8e07a3']
    
    # Violin plot for percentile data
    vplot0 = axs[0].violinplot(median_plot, positions=[1, 2], showmedians=True, showextrema=False)
    for i, body in enumerate(vplot0['bodies']):
        body.set_facecolor(color_scheme[i])
        body.set_edgecolor('black')
        body.set_alpha(0.8)
    for partname in [ 'cmedians']:
        vplot0[partname].set_edgecolor('black')
    
    # Violin plot for cell type data
    vplot1 = axs[1].violinplot(cell_type_plot, positions=[1, 2], showmedians=True, showextrema=False)
    for i, body in enumerate(vplot1['bodies']):
        body.set_facecolor(color_scheme[i])
        body.set_edgecolor('black')
        body.set_alpha(0.8)
    for partname in ['cmedians']:
        vplot1[partname].set_edgecolor('black')
    
    # Mann-Whitney U tests
    _, p_percentile = mannwhitneyu(median_plot[0], median_plot[1], alternative='two-sided')
    _, p_cell_type = mannwhitneyu(cell_type_plot[0], cell_type_plot[1], alternative='two-sided')
    
    # Add significance annotations
    for ax, data, p_val in zip(axs, [median_plot, cell_type_plot], [p_percentile, p_cell_type]):
        y_max = max(np.max(data[0]), np.max(data[1]))
        y_range = y_max - min(np.min(data[0]), np.min(data[1]))
        bar_height = y_max + y_range * 0.05
        bar_tips = bar_height - y_range * 0.02
        text_height = bar_height + y_range * 0.01
        
        ax.plot([1, 1, 2, 2], [bar_tips, bar_height, bar_height, bar_tips], color='black', linewidth=1)
        ax.text(1.5, text_height, get_significance_stars(p_val), ha='center', va='bottom', fontsize=12)
    
    # Set tick labels with sample sizes
    percentile_tick_labels = [f'{label}\nn={median_plot[i].size:,}' for i, label in enumerate(tick_labels)]
    cell_type_tick_labels = [f'{label}\nn={cell_type_plot[i].size:,}' for i, label in enumerate(tick_labels)]
    
    axs[0].set_xticks([1, 2])
    axs[0].set_xticklabels(percentile_tick_labels)
    axs[1].set_xticks([1, 2])
    axs[1].set_xticklabels(cell_type_tick_labels)
    
    for i in range(2):
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    
    axs[0].set_ylabel('Median CpG Methylation Probability')
    axs[1].set_ylabel('Cell Type Methylation STD')
    
    fig_0.tight_layout()
    fig_1.tight_layout()
    fig_0.savefig(PLOT_DIR / 'median_meth_perturb_violin.png')
    fig_0.savefig(PLOT_DIR / 'median_meth_perturb_violin.pdf')

    fig_1.savefig(PLOT_DIR / 'cell_type_iq_perturb_violin.png')
    fig_1.savefig(PLOT_DIR / 'cell_type_iq_perturb_violin.pdf')
   


def get_labelled_cell_type_sample(sample_data):
    sample_data = sample_data[sample_data['successful'] &  (sample_data['penalty']==MAIN_PENALTY)].copy()
    sample_data['to_tumor'] = (sample_data['original_probability']<0.5) & (sample_data['modified_probability']>0.5)
    sample_data['hypo_switch'] = (sample_data['original_methylation']>=0.5) & (sample_data['modified_methylation']<=0.5)
    sample_data['hyper_switch'] = (sample_data['original_methylation']<=0.5) & (sample_data['modified_methylation']>=0.5)
    sample_data['cpg_movement'] = np.sign(sample_data['modified_methylation']-sample_data['original_methylation']).astype(int)
    
    cell_type_cols = [col for col in sample_data.columns if col.startswith('average_methylation_')]
    cell_types = [col.replace('average_methylation_','') for col in cell_type_cols]

    cell_type_mean = np.nanmean(sample_data[cell_type_cols],axis=1)
    sample_data['cell_type_mean'] =cell_type_mean
    extra_data = {}
    for cell_type_index,average_meth_column in enumerate(cell_type_cols):
        cell_type = cell_types[cell_type_index]
        extra_data[f'{cell_type}_marker'] = (sample_data[average_meth_column]-cell_type_mean).abs()>=0.5
        extra_data[f'{cell_type}_total'] = np.sum(extra_data[f'{cell_type}_marker'])
        extra_data[f'{cell_type}_movement'] = np.sign(sample_data[average_meth_column]-cell_type_mean)
    extra_data = pd.DataFrame(extra_data)
    
    sample_data = pd.concat([sample_data,extra_data],axis=1)
    
    return sample_data.reset_index(drop=True).copy()

def get_cell_type_hits(all_samples):

    return_data = []
    for sample_id,sample_data  in all_samples.groupby('sample_id'):
        cancer_type = 'Prostate' if sample_id.startswith('BS') else 'Ovarian'
        sample_data = get_labelled_cell_type_sample(sample_data)
        

        n_samples = len(sample_data)
        sample_markers = sample_data[(sample_data['modified_methylation']-sample_data['original_methylation']).abs()>0.5]


        cell_type_cols = [col for col in sample_data.columns if col.startswith('average_methylation_')]
        cell_types = [col.replace('average_methylation_','') for col in cell_type_cols]


        for cell_type_index,average_meth_column in enumerate(cell_type_cols):
            cell_type = cell_types[cell_type_index]
        
            cell_df = sample_markers[sample_markers[f'{cell_type}_marker']].copy()
            if len(cell_df)<20:
                continue
            
            agree = np.where(cell_df['cpg_movement']==cell_df[f'{cell_type}_movement'].astype(int),1,-1)
            
            test_vector = np.multiply(agree,np.where(cell_df['to_tumor'],1,-1))
            tumor_agree = np.mean(np.where(test_vector<0,0,1))

            marker_rate = len(cell_df)/cell_df[f'{cell_type}_total'].iloc[0]
            row = {'cancer_type':cancer_type,'sample_id':sample_id,'cell_type':cell_type,'tumor_rate':tumor_agree,'marker_rate':marker_rate}
            return_data.append(row)
    
    return pd.DataFrame(return_data)


def get_cell_type_color_scheme(cell_type):
    cell_type_group = get_cell_type_group(cell_type)
    if cell_type_group =='Epithelial':
        return '#9e1dd1'
    if cell_type_group =='Immunue':
        return '#6aad05'
    return '#808080'

def get_cell_type_group(cell_type):
    if 'Epithelial' in cell_type:
        return 'Epithelial'
    if 'Blood' in cell_type or 'Macrophage' in cell_type or 'CNVS' in cell_type:
        return 'Immunue'
    return 'Other'

def add_custom_cell_type_legend(fig):
    #thank you gemini 3
    # Define (Label to display, Dummy string to generate color)
    legend_groups = [
        ('Epithelial Cell Types', 'Epithelial'), # Contains 'Epithelial'
        ('Immune Cell Types',     'Blood'),      # Contains 'Blood' to trigger the 2nd if-statement
        ('Other Cell Types',      'Other')       # Triggers the else statement
    ]
    
    legend_elements = []
    for label, lookup_key in legend_groups:
        # Dynamically get color so it's not defined twice
        color = get_cell_type_color_scheme(lookup_key)
        
        # Create the dot
        element = Line2D([0], [0], marker='o', color='w', label=label,
                         markerfacecolor=color, markersize=8)
        legend_elements.append(element)

    plt.subplots_adjust(right=0.82)

    
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.83, 0.5))
    

def plot_cell_type_hits(sample_data):
    sample_data = get_supplementary_annotations(sample_data)
    sample_data = sample_data.to_pandas()
    cell_type_scores = get_cell_type_hits(sample_data)
    
    
    cell_type_scores['color'] = cell_type_scores['cell_type'].apply(get_cell_type_color_scheme)
    cell_type_scores['cell_type_group'] = cell_type_scores['cell_type'].apply(get_cell_type_group)
    write_to_log('=====CELL TYPE HITS ====')
    write_to_log(cell_type_scores.groupby(['cell_type_group'])[['tumor_rate','marker_rate']].agg(['mean','std']).reset_index().to_string())
    write_to_log(cell_type_scores.groupby(['cancer_type','cell_type_group'])[['tumor_rate','marker_rate']].agg(['mean','std']).reset_index().to_string())
    
    x = cell_type_scores[cell_type_scores['marker_rate']>=0.15].groupby(['cell_type'])['tumor_rate'].agg(['mean','size']).reset_index()
    x = x.rename(columns={'mean':'proportion_tumor_aligned','size':'n_samples'})
    
    fig,axs = plt.subplots(2,3,figsize=(12,5))
    axs = axs.flatten()
    plt_count = 0
    #cheap hack to get them in the correct order
    sample_ids = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    for sample_id in sample_ids:
        sample_data = cell_type_scores[cell_type_scores['sample_id']==sample_id]
        axs[plt_count].scatter(sample_data['marker_rate'],sample_data['tumor_rate'],s=3,color=sample_data['color'],alpha=0.8)
        axs[plt_count].set_xlabel('Proportion of marker CpGs\nthat are perturbed')
        axs[plt_count].set_ylabel('Proportion tumor-like')
        axs[plt_count].set_xlim(0,0.5)
        axs[plt_count].set_ylim(-0.05,1.05)
        axs[plt_count].set_title(plotting_tools.get_sample_mapping()[sample_id.split('_')[0]])
        plt_count +=1
    plt.tight_layout()
    add_custom_cell_type_legend(fig)
    
    plt.savefig(SUPPLEMENTARY_PLOT_DIR / 'cell_type_scores.png')
    plt.savefig(SUPPLEMENTARY_PLOT_DIR / 'cell_type_scores.pdf')
@njit
def get_mean_nearest_neighbour_distance(x,reference_positions,use_reference_positions):
    """Compute distance to nearest 1 for each 1 in binary vector x."""
    # Count ones first
    n_ones = 0
    for i in range(len(x)):
        if x[i] == 1:
            n_ones += 1
    
    if n_ones < 2:
        return np.nan
    
    # Get positions of ones
    positions = np.empty(n_ones, dtype=np.int64)
    idx = 0
    for i in range(len(x)):
        if x[i] == 1:
            positions[idx] = i
            idx += 1
    
    # Compute nearest neighbor distances
    nn_dist = np.empty(n_ones, dtype=np.float64)

    
    for i in range(n_ones):
        if use_reference_positions:
            left = np.inf if i == 0 else reference_positions[positions[i]] - reference_positions[positions[i - 1]]
            right = np.inf if i == n_ones - 1 else reference_positions[positions[i+1]] - reference_positions[positions[i]]
        else:
            left = np.inf if i == 0 else positions[i] - positions[i - 1]
            right = np.inf if i == n_ones - 1 else positions[i + 1] - positions[i]
        nn_dist[i] = min(left, right)
    
    return np.mean(nn_dist)
@njit
def get_mean_nearest_neighbour_distance_permuted(x,reference_positions,use_reference_positions):
    x_permuted = np.random.permutation(x)
    return get_mean_nearest_neighbour_distance(x_permuted,reference_positions,use_reference_positions)

@njit(parallel=True)
def get_mean_nearest_neighbour_distance_distribution(x,reference_positions,use_reference_positions,n_permutations):
    permutation_results = np.zeros(n_permutations)
    for i in prange(n_permutations):
        permutation_results[i]= get_mean_nearest_neighbour_distance_permuted(x,reference_positions,use_reference_positions)
    return permutation_results


def get_permuted_neighbour_p_value(true_neighbour_mean,permuted_neighbour_distribution):
    p_value_left = (np.sum(permuted_neighbour_distribution<=true_neighbour_mean)+1)/(permuted_neighbour_distribution.size+1)
    p_value_right = (np.sum(permuted_neighbour_distribution>=true_neighbour_mean)+1)/(permuted_neighbour_distribution.size+1)
    p_value = min(p_value_left,p_value_right)*2
    p_value = min(p_value,1.0)
    return p_value

def get_neighbourhood_data(sample_data,use_reference_positions):
    
    sample_data = sample_data[sample_data['successful'] & (sample_data['penalty']==MAIN_PENALTY)]
    read_count = 0
    n_total = len(sample_data.drop_duplicates(subset=['sample_id','penalty','read_index']))
    neighbourhood_data = []
    for (sample_id,penalty,read_index),read_data in tqdm(sample_data.groupby(['sample_id','penalty','read_index'])):
        true_neighbour_mean = get_mean_nearest_neighbour_distance(read_data['switched_cpg'].values,read_data['position'].values,use_reference_positions)
        permuted_neighbour_distribution = get_mean_nearest_neighbour_distance_distribution(read_data['switched_cpg'].values,read_data['position'].values,use_reference_positions,5000)
        
        p_value =get_permuted_neighbour_p_value(true_neighbour_mean,permuted_neighbour_distribution)
        
        
        r_value = true_neighbour_mean/np.mean(permuted_neighbour_distribution)
        
        read_count +=1

        row_data = {'sample_id':sample_id,'penalty':penalty,'read_index':read_index}
        row_data['p_value'] = p_value
        row_data['r_value'] = r_value
        neighbourhood_data.append(row_data)
    neighbourhood_data =  pd.DataFrame(neighbourhood_data)
    neighbourhood_data['fdr'] = multipletests(neighbourhood_data['p_value'],alpha=0.05,method='fdr_bh')[1]
    neighbourhood_data['-log2_r_value'] = -np.log2(neighbourhood_data['r_value'])
    neighbourhood_data['r_value_closer'] = neighbourhood_data['r_value']<1
    return neighbourhood_data

def plot_neighbourhood_wrapper(sample_data,read_data,use_reference_positions,fdr_threshold=0.05):
    neighbourhood_data = get_neighbourhood_data(sample_data.to_pandas(),use_reference_positions=use_reference_positions)
    neighbourhood_data = neighbourhood_data.merge(read_data.to_pandas()[['read_index','tumor_predicted_read']])
    
    neighbourhood_data['significant'] = neighbourhood_data['fdr']<fdr_threshold

    write_to_log(f'With use reference positions {use_reference_positions}')
    summary_df = neighbourhood_data.groupby(['tumor_predicted_read'])['r_value_closer'].agg(['mean'])
    write_to_log(summary_df.to_string())


    plot_neighbourhood_analysis_violin(neighbourhood_data,use_reference_positions)

def plot_neighbourhood_analysis_violin(neighbourhood_data,use_reference_positions):
    bool_order =(True,False)
    
    neighbourhood_data = neighbourhood_data[['tumor_predicted_read','-log2_r_value']].dropna()
    r_value_plot = [neighbourhood_data[neighbourhood_data['tumor_predicted_read']==x]['-log2_r_value'].values for x in bool_order]
   
    tick_labels = ['Tumor to\n Non-Tumor','Non-Tumor to\nTumor']
    fig, ax = plt.subplots(1, 1, figsize=(3.65, 3.75))

    
    # Violin plot for percentile data
    vplot = ax.violinplot(r_value_plot, positions=[1, 2], showmedians=True, showextrema=False)
    for i, body in enumerate(vplot['bodies']):
        body.set_facecolor(TYPE_COLOR_SCHEME[i])
        body.set_edgecolor('black')
        body.set_alpha(0.8)
    for partname in [ 'cmedians']:
        vplot[partname].set_edgecolor('black')
    
    # Mann-Whitney U tests
    _, p_r_value = mannwhitneyu(r_value_plot[0], r_value_plot[1], alternative='two-sided')
    write_to_log(f'{use_reference_positions} Tumor predicted read p value {p_r_value}')
    ax.axhline(0,lw=2,color='grey',linestyle='dashed',label='Randomly\nDistributed\nPerturbations')
    
    y_max = max(np.max(r_value_plot[0]), np.max(r_value_plot[1]))
    y_range = y_max - min(np.min(r_value_plot[0]), np.min(r_value_plot[1]))
    bar_height = y_max + y_range * 0.05
    bar_tips = bar_height - y_range * 0.02
    text_height = bar_height + y_range * 0.01
    
    ax.plot([1, 1, 2, 2], [bar_tips, bar_height, bar_height, bar_tips], color='black', linewidth=1)
    ax.text(1.5, text_height, get_significance_stars(p_r_value), ha='center', va='bottom', fontsize=12)
    
    # Set tick labels with sample sizes
    r_value_tick_labels = [f'{label}\nn={r_value_plot[i].size:,}' for i, label in enumerate(tick_labels)]
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(r_value_tick_labels)
   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel('Clustering Coefficient')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    fig.tight_layout()
    
    file_title = f'r_value_violin_use_reference_positions_{use_reference_positions}'
    fig.savefig(PLOT_DIR / f'{file_title}.png')
    fig.savefig(PLOT_DIR / f'{file_title}.pdf')
def plot_neighbourhood_analysis(neighbourhood_data,use_reference_positions,fdr_threshold):
    

    p_cutoff = neighbourhood_data[neighbourhood_data['significant']]['p_value'].max()
    
    label_data = {True:{'color':'#c90a00','label':f'FDR < {fdr_threshold:.2f}'},False:{'color':'#3b3b3b','label':None}}
    fig,ax = plt.subplots(1,1,figsize=(4.5,4))

    for sig,significant_data in neighbourhood_data.groupby('significant'):
        
        ax.scatter(significant_data['r_value'],-np.log10(significant_data['p_value']),s=2,color=label_data[sig]['color'],label=label_data[sig]['label'],alpha=0.3)
    ax.axhline(-np.log10(p_cutoff),lw=2,color='#4f4f4f',linestyle='dashed')
    ax.set_xlabel('Observed / Expected Distance between Perturbed CpGs')
    ax.set_ylabel(r'$-\mathrm{log}_{10}(P)$')
    ax.set_xscale('log') 
    legend = ax.legend()
    for handle in legend.legend_handles:
        handle.set_sizes([20])      # Increase marker size
        handle.set_alpha(1.0)        # Full opacity
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    plot_filename = f'cpg_distance_permutations_reference_positions_{use_reference_positions}'
    plt.savefig(PLOT_DIR / f'{plot_filename}.png')
    plt.savefig(PLOT_DIR / f'{plot_filename}.pdf')

def plot_neighbourhood_analysis_cpg_number(neighbourhood_data,read_data,use_reference_positions,fdr_threshold=0.05):

    n_cpg_data = read_data[['read_index','penalty','n_switch']].copy()
    n_cpg_data['Switch_Plot'] = np.where(n_cpg_data['n_switch']>=7,'7+',n_cpg_data['n_switch'])

    neighbourhood_data = neighbourhood_data.merge(n_cpg_data,how='inner')


    p_cutoff = neighbourhood_data[neighbourhood_data['significant']]['p_value'].max()
   
    fig,ax = plt.subplots(1,1,figsize=(5.5,5))

    for n_cpg,n_cpg_data in neighbourhood_data.groupby('Switch_Plot'):
        
        ax.scatter(n_cpg_data['r_value'],-np.log10(n_cpg_data['p_value']),s=2,label=n_cpg,alpha=0.3)
    ax.axhline(-np.log10(p_cutoff),lw=2,color='#4f4f4f',linestyle='dashed')
    ax.set_xlabel('Observed / Expected Distance between Perturbed CpGs')
    ax.set_xscale('log') 
    ax.set_ylabel(r'$-\mathrm{log}_{10}(P)$')
    legend = ax.legend(title='Number of perturbed CpGs',ncol=2)
    for handle in legend.legend_handles:
        handle.set_sizes([20])      # Increase marker size
        handle.set_alpha(1.0)        # Full opacity
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    plot_filename = f'cpg_distance_permutations_reference_positions_{use_reference_positions}_cpg_number'
    plt.savefig(SUPPLEMENTARY_PLOT_DIR / f'{plot_filename}.png')
    plt.savefig(SUPPLEMENTARY_PLOT_DIR / f'{plot_filename}.pdf')

def plot_read_perturbation(read_data,sample_data):
    
    read_data = (
    read_data
    .filter(
        pl.col("successful"),
        pl.col("n_cpgs").is_between(60, 100),
        pl.col("penalty") == MAIN_PENALTY,
        pl.col("frac_switch").is_between(0.01, 0.1)
    )
    )

    plot_data = (
    sample_data
    .filter(
        pl.col("read_index") == 'm84137_240611_154227_s3/199102432/ccs',
        pl.col("penalty") == MAIN_PENALTY,
    )
    )
    plot_data = plot_data.sort(by='position')
    
    perturbation = np.where((plot_data['original_methylation']-plot_data['modified_methylation']).abs()>0.1,plot_data['modified_methylation'].to_numpy(),np.nan)
    fig,axs = plt.subplots(3,1,figsize=(10,3.4))

    N_REPEATS = 5
    im = axs[0].imshow(np.repeat(plot_data['original_methylation'].to_numpy().reshape(1,-1),N_REPEATS,axis=0),cmap='coolwarm',vmin=0.0,vmax=1.0)
    axs[0].set_title(f'Original Read\nTumor Read Probability {plot_data["original_probability"][0]:.2f}')
    axs[1].imshow(np.repeat(perturbation.reshape(1,-1),N_REPEATS,axis=0),cmap='coolwarm',vmin=0.0,vmax=1.0)
    axs[1].set_title('Change Methylation at Specific CpG Sites')
    axs[2].imshow(np.repeat(plot_data['modified_methylation'].to_numpy().reshape(1,-1),N_REPEATS,axis=0),cmap='coolwarm',vmin=0.0,vmax=1.0)
    axs[2].set_title(f'Perturbed Read\nTumor Read Probability {plot_data["modified_probability"][0]:.2f}')
    
    for i in range(3):
        axs[i].set_yticks([])
    axs[0].set_xticks([])
    axs[1].set_xticks([])

    axs[2].set_xlabel('CpG Site')
    fig.subplots_adjust(right=0.85)
    cbar = fig.colorbar(im, ax=axs.tolist(), location='right', shrink=1.0,aspect=10)
    cbar.set_label('Methylation Probability')
    #plt.tight_layout()
    plt.savefig(PLOT_DIR / 'read_perturb_example.png')
    plt.savefig(PLOT_DIR / 'read_perturb_example.pdf')

if __name__ =="__main__":
    

    current_datetime_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #clear and start new logs
    write_to_log(f'LOGS - {current_datetime_string}',append=False)
    
    sample_ids = ['216_TU','244_TU','264_TU','BS15145_TU','BS14772_TU','053_TU']

    sample_data = load_sample_data(sample_ids)
    
    
    read_data = get_read_data(sample_data)

    
    plot_read_perturbation(read_data,sample_data)
    exit()
    
    plot_supplementary_variance_violin(sample_data)
    
    plot_cell_type_hits(sample_data)
    
    write_to_log('=====Neighborhood analysis======')
    plot_neighbourhood_wrapper(sample_data,read_data,use_reference_positions=True)
    plot_neighbourhood_wrapper(sample_data,read_data,use_reference_positions=False)
    
    plot_frac_switch_by_type(sample_data)
    plot_frac_success_by_type(sample_data)
    
    plot_frac_success_by_type_penalty(sample_data)
    plot_frac_switch_by_type_penalty(sample_data)
    plot_switch_directions_by_type(sample_data)
    plot_switch_directions_by_type_penalty(sample_data)
    

    plot_sample_success_proportions(read_data)




    
    
    
    
    
