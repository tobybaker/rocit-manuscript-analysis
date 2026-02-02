
import matplotlib.pyplot as plt
import plotting_tools
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score

def plot_grouped_bar_chart(data_dict, title,filepath, figsize=(13, 4)):
    samples = list(data_dict.keys())
    keys = list(data_dict[samples[0]].keys())  # Assuming all samples have same keys
    
    # Number of samples and keys
    n_samples = len(samples)
    n_keys = len(keys)
    
    # Set up the bar positions
    bar_width = 1.5 / n_keys  # Width of each individual bar
    group_width = bar_width * n_keys  # Total width of each sample group
    x = np.arange(n_samples) * (group_width + 0.5)  # Sample positions with spacing
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    key_mapping  = {'Methylation Only':'Read Only','With Cell Map':'+ Cell Atlas','With Methylation Distribution':'+ Sample Distribution','Complete Data':'+ Cell Atlas & Sample Distribution'}
    # Create bars for each key
    for i, key in enumerate(keys):
        # Extract values for this key across all samples
        values = [data_dict[sample][key] for sample in samples]
        
        # Calculate position for this group of bars
        offset = (i - n_keys/2 + 0.5) * bar_width
        
        # Create the bars
        bars = ax.bar(x + offset, values, bar_width, label=key_mapping[key])
        
        # Optionally add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height<0.1:
                continue
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel('Test AUC')
    ax.set_title(title)
    ax.set_xticks(x)
    sample_ids = [plotting_tools.get_sample_mapping()[sample] for sample in samples]
    ax.set_xticklabels(sample_ids)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filepath)

def load_auc(path):
    in_df = pl.read_parquet(path)
    return roc_auc_score(in_df['tumor_read'],in_df['tumor_probability'])
def get_auc(sample_id,mode):
    #/053_TU/053_TU_use_cell_map_False_use_sample_distribution_True/train_dataset.parquet
    
    
    if mode =='Complete Data':
        
        base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/main_predictions/')
        sample_dir = base_dir / f'{sample_id}_add_normal_False/train_datasets/'
        path = sample_dir / f'train_{sample_id}_add_normal_False_out_{sample_id}_test_dataset.parquet'
        return load_auc(path)

    main_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/custom_input_predictions')
    sample_dir = main_dir/sample_id
    #different directory if not complete data
    use_cell_map = False
    use_sample_distribution = False 

    if mode =='Methylation Only':
        #change nothing
        pass   
    if mode =='With Cell Map':
        use_cell_map = True

    if mode =='With Methylation Distribution':
        use_sample_distribution = True 
    
    prediction_dir = sample_dir/f'{sample_id}_use_cell_map_{use_cell_map}_use_sample_distribution_{use_sample_distribution}'
    path =prediction_dir/'test_dataset.parquet'
    print(path)
    return load_auc(path)
def load_sample_data():
    all_sample_data ={}
    modes = ['Methylation Only','With Cell Map','With Methylation Distribution','Complete Data']
    for sample_id in ['216_TU', '244_TU', '264_TU', '053_TU','BS14772_TU', 'BS15145_TU']:
        sample_data = {}
        for mode in modes:
            if mode =='Methylation Only':
                sample_data[mode] = 0.5
            else:
                sample_data[mode] = get_auc(sample_id,mode)
        all_sample_data[sample_id.split('_')[0]] = sample_data
    return all_sample_data

def get_aggregated_data(sample_data):
    sample_ids = list(sample_data)
    conditions = list(sample_data[sample_ids[0]])
    
    means = []
    y_err = []
    for condition in conditions:
        condition_data = np.array([sample_data[sample_id][condition]/sample_data[sample_id]['Complete Data'] for sample_id in sample_ids])
        
        print(condition_data)
        mean = np.mean(condition_data)
        std = np.std(condition_data)
        means.append(mean)
        y_err.append(std)
    
    sample_data = [[sample_data[sample_id][condition] for condition in conditions] for sample_id in sample_ids]
    return {'Conditions':conditions,'Means':np.array(means),'Err':np.array(y_err),'Sample_Data':sample_data}


def plot_aggregated_data(aggregated_data,out_path):
    key_mapping  = {'Methylation Only':'Read Only','With Cell Map':'+ Cell Atlas','With Methylation Distribution':'+ Sample Distribution','Complete Data':'+ Cell Atlas & Sample Distribution'}
    color_mapping  = {'Methylation Only':'#FF9970','With Cell Map':'#9BB1FF','With Methylation Distribution':'#C294C7'}
    
    fig,ax = plt.subplots(1,1,figsize=(5,3))

    for index,condition in enumerate(aggregated_data['Conditions']):
        if condition =='Complete Data':
            continue
        bars = ax.bar([index],[aggregated_data['Means'][index]],yerr=[aggregated_data['Err'][index]],capsize=5,label=key_mapping[condition],color=color_mapping[condition])
        for bar in bars:
            height = aggregated_data['Means'][index]+aggregated_data['Err'][index]+0.005
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{aggregated_data["Means"][index]:.3f}', ha='center', va='bottom', fontsize=10)
    
    

    ax.legend(title='Input Data',bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks([])
    ax.set_ylabel('Test AUC / Test AUC on complete data')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    
def plot_custom_training_runs():  

    sample_data = load_sample_data()

    aggregated_data = get_aggregated_data(sample_data)
    #main_path = '../../paper_plots/figure_2/data_input_comparison.png'
    main_path = 'test.png'
    plot_aggregated_data(aggregated_data,main_path)
    #plot_aggregated_data(aggregated_data,main_path.replace('.png','.pdf'))
    
        
    #sup_path = '../../paper_plots/supplementary_figures/data_input_comparison_by_sample.png'
    sup_path = 'test_by_sample.png'
    plot_grouped_bar_chart(sample_data, 'Model performance by input data',sup_path)
    #plot_grouped_bar_chart(sample_data, 'Model performance by input data',sup_path.replace('.png','.pdf'))


if __name__ == "__main__":
    plot_custom_training_runs()