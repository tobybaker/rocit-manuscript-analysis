import polars as pl
import plotting_tools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path
def create_vertical_violins_two_axes(data_dict, sample_ids=None, condition_names=None, 
                                 colors=None, figsize=(8, 4)):
    
    if sample_ids is None:
        sample_ids = list(data_dict.keys())

    if condition_names is None:
        condition_names = list(data_dict[sample_ids[0]].keys())

    if colors is None:
        colors = ['#5496ff','#aa49c4']

    fig, axs = plt.subplots(2,1,figsize=figsize)
    
    width = 0.75 
    sample_positions = np.arange(len(sample_ids))


    for cond_idx, condition in enumerate(condition_names):
        for samp_idx, sample_id in enumerate(sample_ids):
            d = data_dict[sample_id][condition]
    
            
            vp = axs[cond_idx].violinplot([d], positions=[samp_idx], 
                             widths=[width],showextrema=False)
    
            
            vp['bodies'][0].set_facecolor(plotting_tools.get_sample_color_scheme()[sample_id])
            vp['bodies'][0].set_alpha(0.7)
            

    # Customize the plot

    for i in range(2):
        
        if i ==1:
            axs[i].set_xticks(sample_positions)
            axs[i].set_xticklabels(sample_ids)
        else:
            axs[i].set_xticks([])
    
        # Set x-axis limits to ensure all violins are visible
        axs[i].set_xlim(-0.5, len(sample_ids) - 0.5)

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    axs[0].set_title('Training without matched normal reads')
    axs[1].set_title('Training with matched normal reads')
    fig.supylabel('Non-tumor read probability')
    # Add legend
    #legend_elements = [Patch(facecolor=colors[i % len(colors)],alpha=0.7, label=condition_names[i])
    #                  for i in range(len(condition_names))]
    
    #axs[i].legend(handles=legend_elements,bbox_to_anchor=(1.05, 1), loc='upper left',title='Training Data')

    
    plt.tight_layout()
    

    
    return fig, axs
def create_vertical_paired_violins(data_dict, sample_ids=None, condition_names=None, 
                                 colors=None, figsize=(10, 3),legend_title='Training Data'):
    
    if sample_ids is None:
        sample_ids = list(data_dict.keys())

    if condition_names is None:
        condition_names = list(data_dict[sample_ids[0]].keys())

    if colors is None:
        colors = ['#5496ff','#aa49c4']

    fig, ax = plt.subplots(figsize=figsize)
    
    n_conditions = len(condition_names)
    width = 0.75 / n_conditions
    sample_positions = np.arange(len(sample_ids))
    
    # Calculate offset positions for each condition
    # Use smaller offsets and ensure they don't push violins outside plot area
    offset_range =0.2
    offsets = np.linspace(-offset_range, offset_range, n_conditions)

    for cond_idx, condition in enumerate(condition_names):

        for samp_idx, sample_id in enumerate(sample_ids):
            d = data_dict[sample_id][condition]
            
            vp = ax.violinplot([d], positions=[samp_idx + offsets[cond_idx]], 
                             widths=[width],showextrema=False)
    
            
            vp['bodies'][0].set_facecolor(colors[cond_idx])
            vp['bodies'][0].set_alpha(0.7)
            vp['bodies'][0].set_edgecolor('black')

    # Customize the plot
    ax.set_xticks(sample_positions)
    ax.set_xticklabels(sample_ids)
    ax.set_ylabel('Non-Tumor Read Probability')
    ax.set_title('Matched Normal Reads')
    ax.grid(True, alpha=0.3, axis='y')
    
    
    # Set x-axis limits to ensure all violins are visible
    ax.set_xlim(-0.5, len(sample_ids) - 0.5)
    
    # Add legend
    legend_elements = [Patch(facecolor=colors[i % len(colors)],alpha=0.7, label=condition_names[i])
                      for i in range(len(condition_names))]
    
    ax.legend(handles=legend_elements,bbox_to_anchor=(1.05, 1), loc='upper left',title=legend_title)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    

    
    return fig, ax

def write_add_normal_data(average_store):
    no_normal_mean = np.mean(average_store[False])*100.0
    no_normal_std = np.std(average_store[False])*100.0
    normal_mean = np.mean(average_store[True])*100.0
    normal_std = np.std(average_store[True])*100.0

    out_string = f'The average non-tumor read probability across the matched normal cohort was {no_normal_mean:.2f}% (SD = {no_normal_std:.2f}%). Supplementing the model training with a small number of reads from the matched normal as labeled non-tumor reads (Methods) increased the average non-tumor probability for the matched normal reads to {normal_mean:.2f}% (SD = {normal_std:.2f}%)'

    out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/text')
    with open(f'{out_dir}/add_normal_data.txt','w') as out:
        out.write(out_string)
def load_probability_distributions():
    #
    main_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/main_predictions')
    probability_data = {}
    sample_ids = ['216','244','264','053','BS14772','BS15145']
    extra_store = {True:[],False:[]}
    for sample_id in sample_ids:
        sample_data_store = {}
        
        for add_normal_label,add_normal in [('No Matched Normal',False),('Matched Normal',True)]:
            data_dir = main_dir/f'{sample_id}_TU_add_normal_{add_normal}/full_datasets/'
            filename = f'train_{sample_id}_TU_add_normal_{add_normal}_out_{sample_id}_NL_all_reads.parquet'
            sample_data_path = data_dir/filename
            
            sample_data = pl.read_parquet(sample_data_path)
            #sample_data = sample_data.rename({"Sample_ID":'sample_id', "Read_Index":'read_index', "Chromosome":'chromosome', "Tumor_Probability":'tumor_probability'})
            sample_data = sample_data.with_columns(pl.col('chromosome').cast(pl.Categorical))
            training_data_dir = Path('/hot/user/tobybaker/ROCIT_Paper/input_data/labelled_data/')
            training_data_path = training_data_dir/f'{sample_id}_NL_labelled_data.parquet'
            training_data = pl.scan_parquet(training_data_path).select(['chromosome','read_index']).unique()
            training_data = training_data.collect()
             
            sample_data = sample_data.join(training_data,on=['chromosome','read_index'],how='anti')


            
            sample_data_store[add_normal_label] = 1.0-sample_data['tumor_probability'].to_numpy()
            extra_store[add_normal].append(np.mean(sample_data_store[add_normal_label]))
        
        assert sample_data_store['No Matched Normal'].size == sample_data_store['Matched Normal'].size

        data_label = f'{plotting_tools.sample_mapping[sample_id]}\n${sample_data_store["Matched Normal"].size/1e6:.1f} × 10^{6}$ Reads'
        probability_data[data_label] = sample_data_store

    write_add_normal_data(extra_store)
   
    return probability_data





if __name__ =='__main__':
    probability_data = load_probability_distributions()

    print(probability_data)
    for key in probability_data:
        for x in probability_data[key]:
            print(key,x,np.mean(probability_data[key][x]))
    
    fig,ax = create_vertical_paired_violins(probability_data)
    
    plot_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/model_performance')
    plt.savefig(plot_dir/'add_normal_probability_compare.png')
    plt.savefig(plot_dir/'add_normal_probability_compare.pdf')