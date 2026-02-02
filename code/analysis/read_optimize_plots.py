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
MODIFICATION_THRESHOLD = 0.1
PROBABILITY_THRESHOLD = 0.2

MAIN_DIR = Path('/hot/user/tobybaker/CellTypeClassifier/paper_plots')
PLOT_DIR = MAIN_DIR /'read_interpretation'
SUPPLEMENTARY_PLOT_DIR = MAIN_DIR/'supplementary_figures'
LOG_PATH = MAIN_DIR /'latex_tables/read_optimize_out.txt'
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

def bin_probs(prob_series):
    """
    Bin a probability Series into n fixed-width bins.
    Returns a Series of bin labels (floats representing bin centers).
    """
    bins = np.linspace(0,1,11)
    labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(bins.size-1)]
    return pd.cut(prob_series, bins=bins, labels=labels, include_lowest=True)
def get_sequential_colors(x,min_val=0.05,max_val=0.95):
    
    cmap = plt.cm.viridis
    colors = [cmap((i / (x - 1)*(max_val-min_val))+min_val) if x > 1 else cmap(0.5) for i in range(x)]
    return colors
def load_dataframe(filepath):
    in_df = pd.read_parquet(filepath)
    in_df = in_df[in_df['Read_Index'].str.contains('suppFalse')]
    #in_df['Read_Index'] = in_df['Read_Index'].str.partition('|')[0]
    
    in_df = in_df[(in_df['Original_Probability']<PROBABILITY_THRESHOLD) | (in_df['Original_Probability']>1-PROBABILITY_THRESHOLD)].copy()
    in_df['Tumor_Predicted_Read'] = in_df['Original_Probability']>0.5
    sample_id = filepath.split('/')[-2]
    penalty = int(float(re.search(r'penalty_(\d+\.\d+)', filepath).group(1)))
    in_df['Penalty'] = penalty
    in_df['Sample_ID'] = sample_id

    in_df['Original_Bin'] = bin_probs(in_df['Original_Methylation'])
    in_df['Modified_Bin'] = bin_probs(in_df['Modified_Methylation'])

    in_df['Switched_CpG'] = (in_df['Original_Methylation']-in_df['Modified_Methylation']).abs()>MODIFICATION_THRESHOLD

    in_df['Switched_CpG'] = (in_df['Original_Methylation']-in_df['Modified_Methylation']).abs()>MODIFICATION_THRESHOLD
    in_df = in_df.rename(columns={'Positions':'Position'})

    switched_to_non_tumor = np.logical_and(in_df['Tumor_Predicted_Read'],in_df['Modified_Probability']<PROBABILITY_THRESHOLD)
    switched_to_tumor = np.logical_and(np.logical_not(in_df['Tumor_Predicted_Read']),in_df['Modified_Probability']>(1-PROBABILITY_THRESHOLD))
    in_df['Tumor_to_Non_Tumor'] = switched_to_non_tumor
    in_df['Non_Tumor_to_Tumor'] = switched_to_non_tumor
    in_df['Successful'] =np.logical_or(switched_to_non_tumor,switched_to_tumor)
    return in_df.reset_index(drop=True).copy()

def get_read_data(sample_data):

    read_data = sample_data.groupby(['Read_Index','Chromosome','Sample_ID','Penalty','Tumor_Predicted_Read','Original_Probability','Modified_Probability','Successful'])['Switched_CpG'].agg(['sum','size']).reset_index()
    read_data = read_data.rename(columns={'sum':'N_Switch','size':'N_CpGs'})
    read_data['Frac_Switch'] = read_data['N_Switch']/read_data['N_CpGs']
    
    
    return read_data
def load_sample_data(sample_ids):
    in_dir = '/hot/user/tobybaker/CellTypeClassifier/output/read_optimisation_l0_mean_high_lr/'
    
    sample_data = []
    for sample_id in sample_ids:
        sample_dir =f'{in_dir}/{sample_id}'
        
        for filename in os.listdir(sample_dir):
            if not filename.endswith('.parquet'):
                continue
            
            filepath = os.path.join(sample_dir,filename)
            in_df = load_dataframe(filepath)
            sample_data.append(in_df)
        
    return pd.concat(sample_data)


def get_aggregated_success_proportions(read_data):
    read_counts = read_data.groupby(['Penalty'])['Successful'].agg(['size','sum']).reset_index()
    read_counts = read_counts.rename(columns={'size':'N_Observations','sum':'N_Success'})
    read_counts['Proportion'] = read_counts['N_Success']/read_counts['N_Observations']

    ci_low, ci_high = proportion_confint(
    count=read_counts['N_Success'],
    nobs=read_counts['N_Observations'],
    alpha=0.05,  # for 95% CI (1 - 0.05 = 0.95)
    method='wilson'
)
    read_counts['CI_Low'] = ci_low
    read_counts['CI_High'] = ci_high
    return read_counts

def plot_aggregated_success_proportions(read_data,color_scheme):
    read_counts = get_aggregated_success_proportions(read_data)

    write_to_log('====Proportion Success====')
    write_to_log(read_counts.to_string())

    fig,ax = plt.subplots(1,1,figsize=(2,3))

    yerr = (read_counts['Proportion']-read_counts['CI_Low'],read_counts['CI_High']-read_counts['Proportion'])
    ax.bar(np.arange(len(read_counts)),read_counts['Proportion'],color=color_scheme,yerr=yerr,capsize=5)
    ax.set_xticks(np.arange(len(read_counts)))
    ax.set_xticklabels(read_counts['Penalty'])
    ax.set_ylabel('Proportion of reads converted')
    ax.set_xlabel('CpG Modification\nPenalty')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'n_success_reads.png')
    plt.savefig(PLOT_DIR / 'n_success_reads.pdf')


def get_sample_success_proportions(read_data):
    read_counts = read_data.groupby(['Sample_ID','Tumor_Predicted_Read','Penalty'])['Successful'].agg(['size','sum']).reset_index()
    read_counts = read_counts.rename(columns={'size':'N_Observations','sum':'N_Success'})
    read_counts['Proportion'] = read_counts['N_Success']/read_counts['N_Observations']

    ci_low, ci_high = proportion_confint(
    count=read_counts['N_Success'],
    nobs=read_counts['N_Observations'],
    alpha=0.05,  # for 95% CI (1 - 0.05 = 0.95)
    method='wilson'
)
    read_counts['CI_Low'] = ci_low
    read_counts['CI_High'] = ci_high
    return read_counts
def plot_sample_success_proportions(read_data):

    color_scheme = {True:'blue',False:'red'}
    legend_mapping = {True:'Tumor to Non-Tumor',False:"Non-Tumor to Tumor"}
    read_counts = get_sample_success_proportions(read_data)
    
    fig,axs = plt.subplots(2,3,figsize=(10,5))
    axs = axs.flatten()
    sample_count = 0
    
    bar_width = 0.3
    for sample_id,sample_counts in read_counts.groupby('Sample_ID'):
        ax = axs[sample_count]
        tumor_offset = -bar_width
        for tumor_predicted_read,tumor_table in sample_counts.groupby('Tumor_Predicted_Read'):
            
            yerr = (tumor_table['Proportion']-tumor_table['CI_Low'],tumor_table['CI_High']-tumor_table['Proportion'])
            ax.bar(np.arange(len(tumor_table))+tumor_offset,tumor_table['Proportion'],color=color_scheme[tumor_predicted_read],yerr=yerr,capsize=5,label=legend_mapping[tumor_predicted_read],align='edge',width=bar_width)
            tumor_offset += bar_width
        ax.set_xticks(np.arange(read_data['Penalty'].nunique()))
        ax.set_xticklabels(sorted(read_data['Penalty'].unique()))
        ax.set_ylabel('Proportion of reads converted')
        ax.set_xlabel('CpG Modification\nPenalty')
        ax.legend(title='Original Predicition')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(plotting_tools.get_sample_mapping()[sample_id.split('_')[0]])

        sample_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'sample_success_reads.png')
    plt.savefig(PLOT_DIR / 'sample_success_reads.pdf')
def plot_frac_switch(read_data,color_scheme):
    
    success_data = read_data[read_data['Successful']]
    penalties = sorted(success_data['Penalty'].unique())
    penalty_data = [success_data[success_data['Penalty']==penalty]['Frac_Switch'].values for penalty in penalties]

    write_to_log('===-FRAC SWITCH===')
    for index,penalty_values in enumerate(penalty_data):
        log_text = f'Penalty {penalties[index]} - Switch Proportion {np.mean(penalty_values)*100.0:.2f}% STD{np.std(penalty_values)*100.0:.2f}%'
        write_to_log(log_text)
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    
    medianprops = {'color': 'black'}
    flierprops = {'marker': 'o', 'markerfacecolor': 'black', 'markersize': 2,
              'linestyle': 'none', 'alpha': 0.1}
    bplot = ax.boxplot(penalty_data,
                   patch_artist=True,
                   tick_labels=penalties,
                   medianprops=medianprops,
                   flierprops=flierprops)
    
    for patch, color in zip(bplot['boxes'], color_scheme):
        patch.set_facecolor(color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Proportion of perturbed CpGs per read')
    ax.set_xlabel('CpG Modification\nPenalty')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_cpg_switch.png')
    plt.savefig(PLOT_DIR / 'frac_cpg_switch.pdf')

def plot_n_switch(read_data,color_scheme):
    
    success_data = read_data[read_data['Successful']]
    penalties = sorted(success_data['Penalty'].unique())
    penalty_data = [success_data[success_data['Penalty']==penalty]['N_Switch'].values for penalty in penalties]

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    
    medianprops = {'color': 'black'}
    flierprops = {'marker': 'o', 'markerfacecolor': 'black', 'markersize': 2,
              'linestyle': 'none', 'alpha': 0.1}
    bplot = ax.boxplot(penalty_data,
                   patch_artist=True,
                   tick_labels=penalties,
                   medianprops=medianprops,
                   flierprops=flierprops)
    
    for patch, color in zip(bplot['boxes'], color_scheme):
        patch.set_facecolor(color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Number of perturbed CpGs per read')
    ax.set_xlabel('CpG Modification\nPenalty')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'n_cpg_switch.png')
    plt.savefig(PLOT_DIR / 'n_cpg_switch.pdf')



def get_switch_cpg_proportions(sample_data):
    success_data = sample_data[(sample_data['Successful']) & (sample_data['Penalty']==MAIN_PENALTY)]
    switch_counts = success_data.groupby(['Original_Bin','Tumor_Predicted_Read'])['Switched_CpG'].agg(['size','sum']).reset_index()
    switch_counts = switch_counts.rename(columns={'size':'N_Observations','sum':'Switched_CpG'})
    switch_counts['Proportion'] = switch_counts['Switched_CpG']/switch_counts['N_Observations']

    ci_low, ci_high = proportion_confint(
    count=switch_counts['Switched_CpG'],
    nobs=switch_counts['N_Observations'],
    alpha=0.05,  # for 95% CI (1 - 0.05 = 0.95)
    method='wilson'
)
    switch_counts['CI_Low'] = ci_low
    switch_counts['CI_High'] = ci_high
    return switch_counts.copy()

def plot_switch_cpg_proportions(sample_data):
    switch_proportions = get_switch_cpg_proportions(sample_data)
    legend_mapping = {True:'Tumor to Non-Tumor',False:"Non-Tumor to Tumor"}
    color_scheme = {True:'blue',False:'red'}
    fig,ax = plt.subplots(1,1,figsize=(7,4))
    bar_width = 0.3
    
    tumor_offset = -bar_width
    for tumor_predicted_read,tumor_table in switch_proportions.groupby('Tumor_Predicted_Read'):
        
        yerr = (tumor_table['Proportion']-tumor_table['CI_Low'],tumor_table['CI_High']-tumor_table['Proportion'])
        ax.bar(np.arange(len(tumor_table))+tumor_offset,tumor_table['Proportion'],color=color_scheme[tumor_predicted_read],yerr=yerr,capsize=5,label=legend_mapping[tumor_predicted_read],align='edge',width=bar_width)
        tumor_offset += bar_width
    ax.set_xticks(np.arange(switch_proportions['Original_Bin'].nunique()))
    ax.set_xticklabels(sorted(switch_proportions['Original_Bin'].unique()))
    ax.set_ylabel('Proportion of CpGs Perturbed')
    ax.set_xlabel('Original CpG Methylation Probability')
    ax.legend(title='Read')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(PLOT_DIR / 'switch_cpg_proportions_by_probability.png')
    plt.savefig(PLOT_DIR / 'switch_cpg_proportions_by_probability.pdf')

def plot_probability_transition(sample_data):
    success_data = sample_data[(sample_data['Successful']) & (sample_data['Penalty']==MAIN_PENALTY) &(sample_data['Switched_CpG'])]
    
    write_to_log('====PROBABILITY TRANSITION====')
    text_data = success_data.copy()
    text_data['To_Hyper'] = text_data['Original_Methylation'] < text_data['Modified_Methylation']
    text_data['To_Tumor'] = text_data['Original_Probability'] < text_data['Modified_Probability']
    write_to_log(f'Proportion to hyper {text_data['To_Hyper'].mean()*100.0:.2f}%')

    for to_tumor,to_tumor_text_data in text_data.groupby('To_Tumor'):
        write_to_log(f'To Tumor {to_tumor} - Proportion to hyper {to_tumor_text_data['To_Hyper'].mean()*100.0:.2f}%')


    counts = pd.crosstab(success_data['Modified_Bin'],success_data['Original_Bin'])
    #counts = counts/counts.sum(axis=0)
    counts = counts/counts.sum()
    np.fill_diagonal(counts.values, np.nan)
    fig, ax = plt.subplots()
    im = ax.imshow(counts, cmap='plasma',origin='lower') # You can choose other colormaps

    # 4. Add a color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion of Perturbed CpGs", rotation=-90, va="bottom")

    # 5. Set the labels for x and y axes
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns)
    ax.set_yticklabels(counts.index)

    ax.set_xlabel('Original CpG Probability')
    ax.set_ylabel('Modified CpG Probability')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'probability_transition.png')
    plt.savefig(PLOT_DIR / 'probability_transition.pdf')

def plot_frac_success_by_type(sample_data):
    sample_data = sample_data[ (sample_data['Penalty']==MAIN_PENALTY)]
    sample_data = sample_data[['Sample_ID','Read_Index','Tumor_Predicted_Read','Successful']].drop_duplicates()
    
    switch_by_type = sample_data.groupby(['Sample_ID','Tumor_Predicted_Read'])['Successful'].mean().reset_index()
    switch_by_type_agg = switch_by_type.groupby('Tumor_Predicted_Read')['Successful'].agg(['mean','std']).reset_index()
    switch_by_type_agg = switch_by_type_agg.sort_values(by=['Tumor_Predicted_Read'],ascending=False)
    
    fig,ax = plt.subplots(1,1,figsize=(2.8,3))

  
    ax.bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
    ax.set_xticks(np.arange(len(switch_by_type_agg)))

    n_tumor_to_non_tumor = sample_data['Tumor_Predicted_Read'].sum()
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
    sample_data = sample_data[['Penalty','Sample_ID','Read_Index','Tumor_Predicted_Read','Successful']].drop_duplicates()
    fig,ax = plt.subplots(1,3,figsize=(8,3))
    plt_count = 0
    write_to_log('====FRAC SUCCESS BY PENALTY====')
    for penalty,penalty_data in sample_data.groupby('Penalty'):
        switch_by_type = penalty_data.groupby(['Sample_ID','Tumor_Predicted_Read'])['Successful'].mean().reset_index()
        switch_by_type_agg = switch_by_type.groupby('Tumor_Predicted_Read')['Successful'].agg(['mean','std']).reset_index()
        switch_by_type_agg = switch_by_type_agg.sort_values(by=['Tumor_Predicted_Read'],ascending=False)

        write_to_log(f'Penalty {penalty}')
        write_to_log(switch_by_type_agg.to_string())
    
        ax[plt_count].bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
        ax[plt_count].set_xticks(np.arange(len(switch_by_type_agg)))
        n_tumor_to_non_tumor = penalty_data['Tumor_Predicted_Read'].sum()
        non_tumor_to_n_tumor = len(penalty_data)-n_tumor_to_non_tumor

        #ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
        ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
        ax[plt_count].set_ylabel('Proportion of reads converted')
        ax[plt_count].spines['top'].set_visible(False)
        ax[plt_count].spines['right'].set_visible(False)
        ax[plt_count].set_ylim(0,1.05)
        ax[plt_count].set_title(f'Sparsity Penalty\n{penalty}')

        plt_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_success_by_type_penalty.png')
    plt.savefig(PLOT_DIR / 'frac_success_by_type_penalty.pdf')

def plot_frac_switch_by_type_penalty(sample_data):
    sample_data = sample_data[ (sample_data['Successful'])]
    fig,ax = plt.subplots(1,3,figsize=(8,3))
    plt_count = 0

    write_to_log('====FRAC SWITCH BY PENALTY===')
    for penalty,penalty_data in sample_data.groupby('Penalty'):
        switch_by_type = penalty_data.groupby(['Sample_ID','Tumor_Predicted_Read'])['Switched_CpG'].mean().reset_index()
        switch_by_type_agg = switch_by_type.groupby('Tumor_Predicted_Read')['Switched_CpG'].agg(['mean','std']).reset_index()
        switch_by_type_agg = switch_by_type_agg.sort_values(by=['Tumor_Predicted_Read'],ascending=False)
        write_to_log(f'Penalty {penalty}')
        write_to_log(switch_by_type_agg.to_string())
    
        ax[plt_count].bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
        ax[plt_count].set_xticks(np.arange(len(switch_by_type_agg)))
        n_tumor_to_non_tumor = penalty_data['Tumor_Predicted_Read'].sum()
        non_tumor_to_n_tumor = len(penalty_data)-n_tumor_to_non_tumor

        #ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
        ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
        ax[plt_count].set_ylabel('Proportion of CpGs Perturbed')
        ax[plt_count].spines['top'].set_visible(False)
        ax[plt_count].spines['right'].set_visible(False)
        ax[plt_count].set_ylim(0,0.15)
        ax[plt_count].set_title(f'Sparsity Penalty\n{penalty}')

        plt_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_switch_by_type_penalty.png')
    plt.savefig(PLOT_DIR / 'frac_switch_by_type_penalty.pdf')

def plot_frac_switch_by_type(sample_data):
    sample_data = sample_data[ (sample_data['Penalty']==MAIN_PENALTY) & (sample_data['Successful'])]
    switch_by_type = sample_data.groupby(['Sample_ID','Tumor_Predicted_Read'])['Switched_CpG'].mean().reset_index()
    switch_by_type_agg = switch_by_type.groupby('Tumor_Predicted_Read')['Switched_CpG'].agg(['mean','std']).reset_index()
    switch_by_type_agg = switch_by_type_agg.sort_values(by=['Tumor_Predicted_Read'],ascending=False)
    
    fig,ax = plt.subplots(1,1,figsize=(2.8,3))

  
    ax.bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
    ax.set_xticks(np.arange(len(switch_by_type_agg)))
    n_tumor_to_non_tumor = sample_data['Tumor_Predicted_Read'].sum()
    non_tumor_to_n_tumor = len(sample_data)-n_tumor_to_non_tumor

    #ax.set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
    ax.set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
    ax.set_ylabel('Proportion of CpGs Perturbed')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'frac_switch_by_type.png')
    plt.savefig(PLOT_DIR / 'frac_switch_by_type.pdf')

def plot_switch_directions(sample_data):
    sample_data = sample_data[ (sample_data['Penalty']==MAIN_PENALTY) & (sample_data['Successful']) & (sample_data['Switched_CpG'])].copy()
    sample_data['To_Hyper'] = sample_data['Original_Methylation'] < sample_data['Modified_Methylation']
    switch_by_type = sample_data.groupby(['Sample_ID','Tumor_Predicted_Read'])['To_Hyper'].mean().reset_index()
    switch_by_type_agg = switch_by_type.groupby('Tumor_Predicted_Read')['To_Hyper'].agg(['mean','std']).reset_index()
    switch_by_type_agg = switch_by_type_agg.sort_values(by=['Tumor_Predicted_Read'],ascending=False)

    
    
    fig,ax = plt.subplots(1,1,figsize=(2.8,3))

  
    ax.bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
    ax.set_xticks(np.arange(len(switch_by_type_agg)))
    n_tumor_to_non_tumor = sample_data['Tumor_Predicted_Read'].sum()
    non_tumor_to_n_tumor = len(sample_data)-n_tumor_to_non_tumor

    #ax.set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
    ax.set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
    ax.set_ylabel('Proportion of perturbations\nincreasing methylation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'switch_directions_by_type.png')
    plt.savefig(PLOT_DIR / 'switch_directions_by_type.pdf')

def plot_switch_directions_penalty(sample_data):
    sample_data = sample_data[(sample_data['Successful']) & (sample_data['Switched_CpG'])].copy()
    sample_data['To_Hyper'] = sample_data['Original_Methylation'] < sample_data['Modified_Methylation']
    fig,ax = plt.subplots(1,3,figsize=(8,3))
    plt_count = 0
    write_to_log('===Switch_Directions Penalty===')
    for penalty,penalty_data in sample_data.groupby('Penalty'):
        switch_by_type = penalty_data.groupby(['Sample_ID','Tumor_Predicted_Read'])['To_Hyper'].mean().reset_index()
        switch_by_type_agg = switch_by_type.groupby('Tumor_Predicted_Read')['To_Hyper'].agg(['mean','std']).reset_index()
        switch_by_type_agg = switch_by_type_agg.sort_values(by=['Tumor_Predicted_Read'],ascending=False)
        write_to_log(f'Penalty {penalty}')
        write_to_log(switch_by_type_agg.to_string())

        
        ax[plt_count].bar(np.arange(len(switch_by_type_agg)),switch_by_type_agg['mean'],color=TYPE_COLOR_SCHEME,yerr=switch_by_type_agg['std'],capsize=5)
        ax[plt_count].set_xticks(np.arange(len(switch_by_type_agg)))
        n_tumor_to_non_tumor = penalty_data['Tumor_Predicted_Read'].sum()
        non_tumor_to_n_tumor = len(penalty_data)-n_tumor_to_non_tumor
        ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor',f'Non-Tumor to\nTumor'])
        #ax[plt_count].set_xticklabels([f'Tumor to\n Non-Tumor\nn={n_tumor_to_non_tumor:,}',f'Non-Tumor to\nTumor\nn={non_tumor_to_n_tumor:,}'])
        ax[plt_count].set_ylabel('Proportion of perturbations\nadding methylation')
        ax[plt_count].spines['top'].set_visible(False)
        ax[plt_count].spines['right'].set_visible(False)
        ax[plt_count].set_ylim(0,1.05)
        ax[plt_count].set_title(f'Sparsity Penalty\n{penalty}')

        plt_count +=1
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'switch_directions_by_type_penalty.png')
    plt.savefig(PLOT_DIR / 'switch_directions_by_type_penalty.pdf')

def plot_supplementary_variance_violin(sample_data):
    bool_order =(False,True)
    cell_type_cols = [col for col in sample_data.columns if col.startswith('Average_Methylation_')]
    
    sample_data = sample_data[sample_data['Successful'] &  (sample_data['Penalty']==MAIN_PENALTY)]
    median_data = sample_data[['Switched_CpG','Methylation_Percentile_50']]

    median_data = median_data.dropna()
    median_plot = [median_data[median_data['Switched_CpG']==x]['Methylation_Percentile_50'].values for x in bool_order]
    
    cell_type_data = sample_data[['Switched_CpG']].copy()
    cell_type_data['STD'] = np.nanstd(sample_data[cell_type_cols], axis=1)
    cell_type_data = cell_type_data.dropna()
    cell_type_plot = [cell_type_data[cell_type_data['Switched_CpG'] == x]['STD'].values for x in [False, True]]

    write_to_log('==== Cell Type  & Median Data')
    for data_label,dataset in [('Cell_Type_STD',cell_type_plot),('Median',median_plot)]:
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
    sample_data = sample_data[sample_data['Successful'] &  (sample_data['Penalty']==MAIN_PENALTY)].copy()
    sample_data['To_Tumor'] = (sample_data['Original_Probability']<0.5) & (sample_data['Modified_Probability']>0.5)
    sample_data['Hypo_Switch'] = (sample_data['Original_Methylation']>=0.5) & (sample_data['Modified_Methylation']<=0.5)
    sample_data['Hyper_Switch'] = (sample_data['Original_Methylation']<=0.5) & (sample_data['Modified_Methylation']>=0.5)
    sample_data['CpG_Movement'] = np.sign(sample_data['Modified_Methylation']-sample_data['Original_Methylation']).astype(int)

    cell_type_cols = [col for col in sample_data.columns if col.startswith('Average_Methylation_')]
    cell_types = [col.replace('Average_Methylation_','') for col in cell_type_cols]

    cell_type_mean = np.nanmean(sample_data[cell_type_cols],axis=1)
    sample_data['Cell_Type_Mean'] =cell_type_mean
    extra_data = {}
    for cell_type_index,average_meth_column in enumerate(cell_type_cols):
        cell_type = cell_types[cell_type_index]
        extra_data[f'{cell_type}_Marker'] = (sample_data[average_meth_column]-cell_type_mean).abs()>=0.5
        extra_data[f'{cell_type}_Total'] = np.sum(extra_data[f'{cell_type}_Marker'])
        extra_data[f'{cell_type}_Movement'] = np.sign(sample_data[average_meth_column]-cell_type_mean)
    extra_data = pd.DataFrame(extra_data)
    sample_data = pd.concat([sample_data,extra_data],axis=1)
    
    return sample_data.reset_index(drop=True).copy()

def get_cell_type_hits(all_samples):

    return_data = []
    for sample_id,sample_data  in all_samples.groupby('Sample_ID'):
        cancer_type = 'Prostate' if sample_id.startswith('BS') else 'Ovarian'
        sample_data = get_labelled_cell_type_sample(sample_data)
        n_samples = len(sample_data)
        sample_markers = sample_data[(sample_data['Modified_Methylation']-sample_data['Original_Methylation']).abs()>0.5]
        base_rate = len(sample_markers)/n_samples

        cell_type_cols = [col for col in sample_data.columns if col.startswith('Average_Methylation_')]
        cell_types = [col.replace('Average_Methylation_','') for col in cell_type_cols]
        cell_type_mean = np.nanmean(sample_data[cell_type_cols],axis=1)


        for cell_type_index,average_meth_column in enumerate(cell_type_cols):
            cell_type = cell_types[cell_type_index]
        
            cell_df = sample_markers[sample_markers[f'{cell_type}_Marker']].copy()
            if len(cell_df)<20:
                continue
            
            agree = np.where(cell_df['CpG_Movement']==cell_df[f'{cell_type}_Movement'].astype(int),1,-1)
            
            test_vector = np.multiply(agree,np.where(cell_df['To_Tumor'],1,-1))
            tumor_agree = np.mean(np.where(test_vector<0,0,1))

            marker_rate = len(cell_df)/cell_df[f'{cell_type}_Total'].iloc[0]
            row = {'Cancer_Type':cancer_type,'Sample_ID':sample_id,'Cell_Type':cell_type,'Tumor_Rate':tumor_agree,'Marker_Rate':marker_rate}
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
    cell_type_scores = get_cell_type_hits(sample_data)
    
    cell_type_scores['Color'] = cell_type_scores['Cell_Type'].apply(get_cell_type_color_scheme)
    cell_type_scores['Cell_Type_Group'] = cell_type_scores['Cell_Type'].apply(get_cell_type_group)
    write_to_log('=====CELL TYPE HITS ====')
    write_to_log(cell_type_scores.groupby(['Cell_Type_Group'])[['Tumor_Rate','Marker_Rate']].agg(['mean','std']).reset_index().to_string())
    write_to_log(cell_type_scores.groupby(['Cancer_Type','Cell_Type_Group'])[['Tumor_Rate','Marker_Rate']].agg(['mean','std']).reset_index().to_string())
    
    x = cell_type_scores[cell_type_scores['Marker_Rate']>=0.15].groupby(['Cell_Type'])['Tumor_Rate'].agg(['mean','size']).reset_index()
    x = x.rename(columns={'mean':'Proportion_Tumor_Aligned','size':'N_Samples'})
    
    fig,axs = plt.subplots(2,3,figsize=(12,5))
    axs = axs.flatten()
    plt_count = 0
    #cheap hack to get them in the correct order
    sample_ids = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    for sample_id in sample_ids:
        sample_data = cell_type_scores[cell_type_scores['Sample_ID']==sample_id]
        axs[plt_count].scatter(sample_data['Marker_Rate'],sample_data['Tumor_Rate'],s=3,color=sample_data['Color'],alpha=0.8)
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
    
    sample_data = sample_data[sample_data['Successful'] & (sample_data['Penalty']==MAIN_PENALTY)]
    read_count = 0
    n_total = len(sample_data.drop_duplicates(subset=['Sample_ID','Penalty','Read_Index']))
    neighbourhood_data = []
    for (sample_id,penalty,read_index),read_data in tqdm(sample_data.groupby(['Sample_ID','Penalty','Read_Index'])):
        true_neighbour_mean = get_mean_nearest_neighbour_distance(read_data['Switched_CpG'].values,read_data['Position'].values,use_reference_positions)
        permuted_neighbour_distribution = get_mean_nearest_neighbour_distance_distribution(read_data['Switched_CpG'].values,read_data['Position'].values,use_reference_positions,5000)
        
        p_value =get_permuted_neighbour_p_value(true_neighbour_mean,permuted_neighbour_distribution)
        
        
        r_value = true_neighbour_mean/np.mean(permuted_neighbour_distribution)
        
        read_count +=1

        row_data = {'Sample_ID':sample_id,'Penalty':penalty,'Read_Index':read_index}
        row_data['P_Value'] = p_value
        row_data['R_Value'] = r_value
        neighbourhood_data.append(row_data)
    neighbourhood_data =  pd.DataFrame(neighbourhood_data)
    neighbourhood_data['FDR'] = multipletests(neighbourhood_data['P_Value'],alpha=0.05,method='fdr_bh')[1]
    neighbourhood_data['-Log2_R_Value'] = -np.log2(neighbourhood_data['R_Value'])
    neighbourhood_data['R_Value_Closer'] = neighbourhood_data['R_Value']<1
    return neighbourhood_data

def plot_neighbourhood_wrapper(sample_data,read_data,use_reference_positions,fdr_threshold=0.05):
    neighbourhood_data = get_neighbourhood_data(sample_data,use_reference_positions=use_reference_positions)
    neighbourhood_data = neighbourhood_data.merge(read_data[['Read_Index','Tumor_Predicted_Read']])
    
    neighbourhood_data['Significant'] = neighbourhood_data['FDR']<fdr_threshold
    proportion_significant = neighbourhood_data['Significant'].mean()
    proportion_significant_closer = (neighbourhood_data[neighbourhood_data['Significant']]['R_Value']<1).mean()
    
    write_to_log(f'With use reference positions {use_reference_positions}')
    summary_df = neighbourhood_data.groupby(['Tumor_Predicted_Read'])['R_Value_Closer'].agg(['mean'])
    write_to_log(summary_df.to_string())


    plot_neighbourhood_analysis_violin(neighbourhood_data,use_reference_positions)
    #plot_neighbourhood_analysis_cpg_number(neighbourhood_data,read_data,use_reference_positions,fdr_threshold)

def plot_neighbourhood_analysis_violin(neighbourhood_data,use_reference_positions):
    bool_order =(True,False)
    cell_type_cols = [col for col in sample_data.columns if col.startswith('Average_Methylation_')]
    
    neighbourhood_data = neighbourhood_data[['Tumor_Predicted_Read','-Log2_R_Value']].dropna()
    r_value_plot = [neighbourhood_data[neighbourhood_data['Tumor_Predicted_Read']==x]['-Log2_R_Value'].values for x in bool_order]
   
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
    write_to_log(f'{use_reference_positions} tumor predicted read p value {p_r_value}')
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
    

    p_cutoff = neighbourhood_data[neighbourhood_data['Significant']]['P_Value'].max()
    
    label_data = {True:{'color':'#c90a00','label':f'FDR < {fdr_threshold:.2f}'},False:{'color':'#3b3b3b','label':None}}
    fig,ax = plt.subplots(1,1,figsize=(4.5,4))

    for sig,significant_data in neighbourhood_data.groupby('Significant'):
        
        ax.scatter(significant_data['R_Value'],-np.log10(significant_data['P_Value']),s=2,color=label_data[sig]['color'],label=label_data[sig]['label'],alpha=0.3)
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

    n_cpg_data = read_data[['Read_Index','Penalty','N_Switch']].copy()
    n_cpg_data['Switch_Plot'] = np.where(n_cpg_data['N_Switch']>=7,'7+',n_cpg_data['N_Switch'])

    neighbourhood_data = neighbourhood_data.merge(n_cpg_data,how='inner')


    p_cutoff = neighbourhood_data[neighbourhood_data['Significant']]['P_Value'].max()
   
    fig,ax = plt.subplots(1,1,figsize=(5.5,5))

    for n_cpg,n_cpg_data in neighbourhood_data.groupby('Switch_Plot'):
        
        ax.scatter(n_cpg_data['R_Value'],-np.log10(n_cpg_data['P_Value']),s=2,label=n_cpg,alpha=0.3)
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
    read_data = read_data.copy()
    read_data = read_data[read_data['Successful']]
    read_data = read_data[read_data['N_CpGs'].between(60,100)]
    read_data = read_data[read_data['Penalty']==MAIN_PENALTY]
    read_data = read_data[read_data['Frac_Switch'].between(0.01,0.1)]


    test = sample_data[sample_data['Read_Index'].isin(read_data['Read_Index']) & (sample_data['Penalty']==MAIN_PENALTY)]
    test = test[test['Switched_CpG']].copy()
    test['Mod_Direction'] = np.sign(test['Original_Methylation']-test['Modified_Methylation']).astype(int)==1
    

    plot_data = sample_data[(sample_data['Read_Index'].str.contains('m84209_250513_225956_s2/97060973/ccs')) & (sample_data['Penalty']==MAIN_PENALTY)]
    plot_data = plot_data.sort_values(by='Position')
    
    perturbation = np.where((plot_data['Original_Methylation']-plot_data['Modified_Methylation']).abs()>0.1,plot_data['Modified_Methylation'].values,np.nan)
    fig,axs = plt.subplots(3,1,figsize=(10,2),constrained_layout=True)

    im = axs[0].imshow(plot_data['Original_Methylation'].values.reshape(1,-1),cmap='coolwarm',aspect='auto',vmin=0.0,vmax=1.0)
    axs[0].set_title(f'Original Read\nTumor Read Probability {plot_data["Original_Probability"].iloc[0]:.2f}')
    axs[1].imshow(perturbation.reshape(1,-1),cmap='coolwarm',aspect='auto',vmin=0.0,vmax=1.0)
    axs[1].set_title('Change Methylation at Specific CpG Sites')
    axs[2].imshow(plot_data['Modified_Methylation'].values.reshape(1,-1),cmap='coolwarm',aspect='auto',vmin=0.0,vmax=1.0)
    axs[2].set_title(f'Perturbed Read\nTumor Read Probability {plot_data["Modified_Probability"].iloc[0]:.2f}')
    
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

def plot_read_perturbation_del(read_data,sample_data):
    read_data = read_data.copy()
    read_data = read_data[read_data['Successful']]
    read_data = read_data[read_data['N_CpGs'].between(60,100)]
    read_data = read_data[read_data['Penalty']==MAIN_PENALTY]
    read_data = read_data[read_data['Frac_Switch'].between(0.01,0.1)]


    test = sample_data[sample_data['Read_Index'].isin(read_data['Read_Index']) & (sample_data['Penalty']==MAIN_PENALTY)]
    test = test[test['Switched_CpG']].copy()
    test['Mod_Direction'] = np.sign(test['Original_Methylation']-test['Modified_Methylation']).astype(int)==1
    

    plot_data = sample_data[(sample_data['Read_Index'].str.contains('m84209_250513_225956_s2/97060973/ccs')) & (sample_data['Penalty']==MAIN_PENALTY)]
    plot_data = plot_data.sort_values(by='Position')
    
    perturbation = np.where((plot_data['Original_Methylation']-plot_data['Modified_Methylation']).abs()>0.1,plot_data['Modified_Methylation'].values,np.nan)
    fig,axs = plt.subplots(3,1,figsize=(10,2),constrained_layout=True)

    im = axs[0].imshow(plot_data['Original_Methylation'].values.reshape(1,-1),cmap='coolwarm',aspect='auto',vmin=0.0,vmax=1.0)
    axs[0].set_title(f'Original Read\nTumor Read Probability {plot_data["Original_Probability"].iloc[0]:.2f}')
    axs[1].imshow(perturbation.reshape(1,-1),cmap='coolwarm',aspect='auto',vmin=0.0,vmax=1.0)
    axs[1].set_title('Change Methylation at Specific CpG Sites')
    axs[2].imshow(plot_data['Modified_Methylation'].values.reshape(1,-1),cmap='coolwarm',aspect='auto',vmin=0.0,vmax=1.0)
    axs[2].set_title(f'Perturbed Read\nTumor Read Probability {plot_data["Modified_Probability"].iloc[0]:.2f}')
    
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
    plot_frac_success_by_type_penalty(sample_data)
    plot_frac_switch_by_type_penalty(sample_data)
    plot_switch_directions(sample_data)
    plot_switch_directions_penalty(sample_data)
    

    
    
    
    
    plot_probability_transition(sample_data)
    
    plot_supplementary_variance_violin(sample_data)
    

    color_scheme = get_sequential_colors(sample_data['Penalty'].nunique())
    
    
    
    
    
   
    plot_aggregated_success_proportions(read_data,color_scheme)
    plot_sample_success_proportions(read_data)
    plot_n_switch(read_data,color_scheme)
    plot_frac_switch(read_data,color_scheme)
    plot_frac_success_by_type(sample_data)
    plot_switch_cpg_proportions(sample_data)
    
    
    
    
    
