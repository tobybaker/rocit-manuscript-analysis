import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import plotting_tools
from plot_classification_performance import load_sample_predictions
from pathlib import Path
def plot_grouped_bar_chart(data_dict,filepath, figsize=(8, 4)):
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
    
    # Create bars for each key
    for i, key in enumerate(keys):
        # Extract values for this key across all samples
        values = [data_dict[sample][key] for sample in samples]
        
        # Calculate position for this group of bars
        offset = (i - n_keys/2 + 0.5) * bar_width
        
        # Create the bars
        bars = ax.bar(x + offset, values, bar_width, label=key)
        
        # Optionally add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height<0.1:
                continue
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=6)
    

    ax.set_ylabel('Test Set AUC')
    
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    plt.tight_layout()
    plt.savefig(filepath)

def get_transformer_test_result(sample_id):
    in_df = load_sample_predictions(sample_id,sample_id,add_normal=False,mode='test')
    return roc_auc_score(in_df['tumor_read'],in_df['tumor_probability'])

def get_xgboost_results(sample_id:str):
    in_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/xgboost')
    in_path = in_dir/f'{sample_id}_TU_xgboost_results.tsv'
    in_df = pl.read_csv(in_path,separator="\t")

    auc = in_df['auc'][0]
    return auc

def get_sample_data():
    sample_ids = ['216','244','264','053','BS14772','BS15145']
    sample_data_store = {}
    for sample_id in sample_ids:
        display_id = plotting_tools.get_sample_mapping()[sample_id]
        sample_data_store[display_id] = {}
        sample_data_store[display_id]['XGBoost'] = get_xgboost_results(sample_id)
        sample_data_store[display_id]['Transformer'] = get_transformer_test_result(sample_id)
    return sample_data_store
def write_xgboost_out_data(sample_data):
    out_info = []
    for performance in sample_data.values():
        out_info.append(performance['Transformer']-performance['XGBoost'])
    
    out_string = f'Although good performance with XGBoost was obtained (Fig. ....) it was not as high as the transformer-based model, with an average decrease in AUC for each sample of {np.median(out_info)}.'
    out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/text')
    with open(out_dir/'xgboost_auc.txt','w') as out:
        out.write(out_string)
def plot_xgboost_transformer_classification():
    sample_data = get_sample_data()
    write_xgboost_out_data(sample_data)
    plot_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/supplementary_figures')
    out_path = plot_dir/'xgboost_transformer_compare.png'
    plot_grouped_bar_chart(sample_data,out_path)
    plot_grouped_bar_chart(sample_data,str(out_path).replace('.png','.pdf'))

if __name__ =='__main__':
    plot_xgboost_transformer_classification()