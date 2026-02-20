import polars as pl
import os
import matplotlib.pyplot as plt
import plotting_tools

from pathlib import Path

from sklearn.metrics import roc_auc_score
def get_test_auc(in_path):
    in_df = pl.read_csv(in_path,separator="\t")
    return in_df['AUC_test'].iloc[in_df['AUC_val'].idxmax()]
def load_data(sample_ids):
    read_lengths = [150,500,1000,2500,5000,7500,10000,12500,15000]
    base_dir = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/length_predictions')
    data_store = []
    for sample_id in sample_ids:
        sample_dir = base_dir/sample_id
        for read_length in read_lengths:

            filename = f'train_{sample_id}_read_length_{read_length}_out_{sample_id}_test_dataset.parquet'
            filepath = sample_dir/filename
            
            in_df = pl.read_parquet(filepath)
        
            test_auc = roc_auc_score(in_df['tumor_read'],in_df['tumor_probability'])

            data_store.append({'sample_id':plotting_tools.get_sample_mapping()[sample_id.split('_')[0]],'read_length':read_length,'auc':test_auc})
    return pl.DataFrame(data_store).sort(['sample_id','read_length'])

def get_read_length_data_summary(in_data,length):
    in_data_length = in_data.filter(pl.col('read_length')==length)
    mean = in_data_length['auc'].mean()
    range = in_data_length['auc'].min(),in_data_length['auc'].max()
    return mean,range
def write_length_threshold_data(in_data):
    read_150_mean,read_150_range = get_read_length_data_summary(in_data,150)
    read_15000_mean,read_15000_range = get_read_length_data_summary(in_data,15000)
    out_str = f'AUC for 150bp {read_150_mean:.4f} (Range {read_150_range[0]:.4f}-{read_150_range[1]:.4f}) '
    out_str += f'AUC for 15000bp {read_15000_mean:.4f} (Range {read_15000_range[0]:.4f}-{read_15000_range[1]:.4f})'

    out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/text')
    out_path = out_dir/'length_threshold_auc.txt'
    with open(out_path,'w') as out:
        out.write(out_str)
if __name__ =='__main__':
    sample_ids = ['BS14772_TU','BS15145_TU','216_TU','244_TU','264_TU','053_TU']
    in_data = load_data(sample_ids)
    
    write_length_threshold_data(in_data)
    
    
    fig,ax = plt.subplots(1,1,figsize=(6,3.5))
    for sample_id,sample_data in in_data.group_by('sample_id'):
        sample_id = sample_id[0]
        sample_data = sample_data.sort('read_length')
        ax.plot(sample_data['read_length'],sample_data['auc'],label=sample_id,linestyle='dashed',marker='o',color=plotting_tools.get_sample_color_scheme()[sample_id])
        
    ax.legend()
    ax.set_xlabel('Read Length')
    ax.set_ylabel('Test AUC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_dir = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/plots/model_performance')
    plt.savefig(out_dir/'read_length_performance_thresholds.png')
    plt.savefig(out_dir /'read_length_performance_thresholds.pdf')
    