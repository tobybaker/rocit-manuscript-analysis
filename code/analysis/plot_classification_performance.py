import polars as pl
import numpy as np
import plotting_tools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick

from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score

import latex_page_maker
import plot_custom_training_runs

from pathlib import Path


BASE_PREDICTION_DIR = Path('/hot/user/tobybaker/ROCIT_Paper/predictions/main_predictions/')
BASE_OUT_DIR = Path('/hot/user/tobybaker/ROCIT_Paper/out_paper/')
SUPPLEMENTARY_FIGURE_DIR = BASE_OUT_DIR /'plots/supplementary_figures'
LATEX_TABLE_DIR = BASE_OUT_DIR /'text'
MAIN_FIGURE_DIR = BASE_OUT_DIR/'plots/model_performance'

def get_sample_cancer_type(sample_id:str):
    if '_' in sample_id:
        sample_id = sample_id.split('_')[0]
    
    ovarian_samples = ['053','192','216','244','264']
    prostate_samples = ['BS14772','BS15145']
    cancer_type_mapping = {sample_id:'Ovarian' for sample_id in ovarian_samples}
    cancer_type_mapping.update({sample_id:'Prostate' for sample_id in prostate_samples})
    
    return cancer_type_mapping[sample_id]
def plot_single_bar_chart(data_dict,filepath, figsize=(15, 4)
                           ,x_label='',y_label='',plt_title='',color_scheme=None):
    samples = list(data_dict.keys())
    
    # Number of samples and keys
    n_samples = len(samples)
    
    # Set up the bar positions
    bar_width = 0.7  # Width of each individual bar

    x = np.arange(n_samples)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    

    # Create the bars
    
    for i,sample in enumerate(samples):
        color = '#4765d1' if color_scheme is None else color_scheme[sample]
        bars = ax.bar(i, data_dict[sample], bar_width,color=color)

        for bar in bars:
            height = bar.get_height()
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        

    ax.set_ylabel(y_label)
    ax.set_title(plt_title)
    ax.set_xticks(x)
    
    ax.set_xticklabels([plotting_tools.sample_mapping[s] for s in samples])
    print([plotting_tools.sample_mapping[s] for s in samples])
    ax.set_xlabel(x_label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.savefig(str(filepath).replace('.png','.pdf'))
def plot_grouped_bar_chart(data_dict,filepath, figsize=(15, 4)
                           ,x_label='',y_label='',plt_title='',legend_title='',color_scheme=None):
    samples = list(data_dict.keys())
    keys = list(data_dict[samples[0]].keys())  # Assuming all samples have same keys
    
    # Number of samples and keys
    n_samples = len(samples)
    n_keys = len(keys)
    use_legend = len(keys)>1
    
    # Set up the bar positions
    bar_width = 1.5/n_keys
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
        color = None if color_scheme is None else color_scheme[key]

        if key in plotting_tools.sample_mapping:
            label = plotting_tools.sample_mapping[key]
        else:
            label = key
        bars = ax.bar(x + offset, values, bar_width, label=label,color=color)
        
        # Optionally add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height<0.1:
                continue
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    

    ax.set_ylabel(y_label)
    ax.set_title(plt_title)
    ax.set_xticks(x)
    ax.set_xticklabels([plotting_tools.sample_mapping[s] for s in samples])
    ax.set_xlabel(x_label)
    if use_legend:
        ax.legend(title=legend_title)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    plt.tight_layout()
    plt.savefig(filepath)
    plt.savefig(str(filepath).replace('.png','.pdf'))
    plt.close(fig)
def process_read_paths(paths):
    read_data  =[]
    for path in paths:
        in_df = pl.read_parquet(path).select(['Read_Index','tumor_read']).unique()
        read_data.append(in_df)
    read_data = pl.concat(read_data).reset_index(drop=True)
    read_data['tumor_read'] = read_data['tumor_read'].astype(int)
    return read_data

def load_sample_predictions(model_sample_id:str,out_sample_id:str,add_normal:bool,mode:str):
    
    training_dir = BASE_PREDICTION_DIR/f'{model_sample_id}_TU_add_normal_{add_normal}/train_datasets'
    filename = f'train_{model_sample_id}_TU_add_normal_{add_normal}_out_{out_sample_id}_TU_{mode}_dataset.parquet'

    in_path = training_dir/filename
    predictions = pl.read_parquet(in_path)
    #predictions = predictions.rename({"Sample_ID":"sample_id", "Read_Index":"read_index", "Chromosome":"chromosome", "Tumor_Probability":"tumor_probability", "Tumor_Read":"tumor_read"})
    predictions = predictions.with_columns((pl.col('tumor_probability')>=0.5).alias('tumor_assignment'))
    return predictions


def load_sample_data(sample_ids):
    
    modes = ['train','test','val']
    sample_ids = ['216','244','264','053','BS14772','BS15145']
    prediction_store = []
    summary_store = []
    
    for out_sample_id in sample_ids:

        for model_sample_id in sample_ids:
            for mode in modes:
                for add_normal in [True,False]:
                    predictions = load_sample_predictions(model_sample_id,out_sample_id,add_normal,mode)
    
                    predictions = predictions.with_columns(
                        pl.lit(out_sample_id.split('_')[0]).alias('out_sample_id'),
                        pl.lit(model_sample_id.split('_')[0]).alias('model_sample_id'),
                        pl.lit(add_normal).alias('add_normal'),
                        pl.lit(mode.capitalize()).alias('mode'),
                    )
                    
                    
                    prediction_store.append(predictions)

                    data_summary = {'out_sample_id':out_sample_id.split('_')[0],'model_sample_id':model_sample_id.split('_')[0],'add_normal':add_normal,'mode':mode.capitalize()}
                    data_summary['AUC'] = roc_auc_score(predictions['tumor_read'],predictions['tumor_probability'])
                    data_summary['MCC'] = matthews_corrcoef(predictions['tumor_read'],predictions['tumor_assignment'])
                    data_summary['F1'] = f1_score(predictions['tumor_read'],predictions['tumor_assignment'])
                    summary_store.append(data_summary)
                    #print(out_sample_id,model_sample_id,add_normal,mode)
    
    summary_store = pl.DataFrame(summary_store)
    sample_enum = pl.Enum(sample_ids)
    categorical_enum = pl.Enum(['Train','Validation','Test'])
    
    summary_store = summary_store.with_columns(
        pl.col('out_sample_id').cast(sample_enum),
        pl.col('model_sample_id').cast(sample_enum),
        pl.col('mode').str.replace('Val','Validation').cast(categorical_enum)
    )


    prediction_store = pl.concat(prediction_store)
    prediction_store = prediction_store.with_columns(
        pl.col('out_sample_id').cast(sample_enum),
        pl.col('model_sample_id').cast(sample_enum),
        pl.col('mode').str.replace('Val','Validation').cast(categorical_enum)
    )
    return prediction_store,summary_store

def write_main_figure_text(main_figure_data_auc):
    print(main_figure_data_auc)
    mean_auc = np.mean(list(main_figure_data_auc.values()))

    out_path = LATEX_TABLE_DIR /'main_figure_text.txt'
    out_text = f'The model demonstrated excellent discriminatory power on our test set across all samples, with a mean area under the receiver operating characteristic curve (AUC) of {mean_auc:.4}'
    with open(out_path,'w') as out:
        
        out.write(out_text)
def plot_main_figure(sample_data_summary):
    main_figure_data = sample_data_summary.filter(pl.col('out_sample_id')==pl.col('model_sample_id'))
    main_figure_data = main_figure_data.filter(~pl.col('add_normal'))

    main_figure_data = main_figure_data.filter(pl.col('mode')=='Test')
    
    main_figure_data_auc = main_figure_data.to_pandas().set_index('out_sample_id')['AUC'].to_dict()
    write_main_figure_text(main_figure_data_auc)
 
    plot_single_bar_chart(main_figure_data_auc,MAIN_FIGURE_DIR/'test_auc_main_samples.png',figsize=(5.5,3),color_scheme=plotting_tools.sample_color_scheme,y_label='Test AUC')

def plot_dataset_distributions(sample_data_summary):
    figure_data = sample_data_summary.filter(pl.col('out_sample_id')==pl.col('model_sample_id'))
    figure_data = figure_data.filter(~pl.col('add_normal'))
    
    for metric in ['AUC','MCC','F1']:
        metric_data =figure_data.to_pandas().set_index(['mode','out_sample_id'])[metric].unstack(fill_value=0).to_dict()
        plot_grouped_bar_chart(metric_data,SUPPLEMENTARY_FIGURE_DIR/f'data_split_distribution_{metric}.png',y_label=metric,legend_title='Dataset',figsize=(8,4))

def plot_add_normal_distributions(sample_data_summary):
    figure_data = sample_data_summary.filter(pl.col('out_sample_id')==pl.col('model_sample_id'))
    figure_data = figure_data.filter(pl.col('mode')=='Test')

    figure_data = figure_data.with_columns(
    pl.when(pl.col("add_normal"))
    .then(pl.lit("With Matched Normal"))
    .otherwise(pl.lit("Without Matched Normal"))
    .alias("add_normal")
    )
    
    for metric in ['AUC','MCC','F1']:
        
        metric_data =figure_data.to_pandas().set_index(['add_normal','out_sample_id'])[metric].unstack(fill_value=0).to_dict()
        plot_grouped_bar_chart(metric_data,SUPPLEMENTARY_FIGURE_DIR/f'add_normal_distribution_{metric}.png',y_label=F'Test {metric}',legend_title='Dataset',figsize=(10,4))


def plot_heatmap(metric_data,metric,filepath):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot with imshow
    im = ax.imshow(metric_data, cmap='YlGn', aspect='auto')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=f'Test {metric}')

    for i in range(len(metric_data.index)):
        for j in range(len(metric_data.columns)):
            value = metric_data.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color='black', fontweight='bold')

    # Set labels and title
    ax.set_xlabel('Testing Sample')
    ax.set_ylabel('Training Sample')

    # Optional: Add tick labels if you want to show the actual X,Y values
    ax.set_xticks(range(len(metric_data.columns)))
    ax.set_xticklabels(metric_data.columns)
    ax.set_yticks(range(len(metric_data.index)))
    ax.set_yticklabels(metric_data.index)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.savefig(str(filepath).replace('.png','.pdf'))

def plot_different_sample_heatmap(sample_data_summary):
    figure_data = sample_data_summary[sample_data_summary['mode']=='Test']
    figure_data = figure_data[~figure_data['add_normal']]   
    figure_data = figure_data.copy()
    figure_data['model_sample_id'] = figure_data['model_sample_id'].map(plotting_tools.sample_mapping)
    figure_data['out_sample_id'] = figure_data['out_sample_id'].map(plotting_tools.sample_mapping)
    
    for metric in ['AUC','MCC','F1']:
        metric_data =figure_data.pivot(index='model_sample_id',columns='out_sample_id',values=metric)
        
        plot_heatmap(metric_data,metric,SUPPLEMENTARY_FIGURE_DIR/f'different_sample_distribution_heatmap_{metric}.png')


def get_different_sample_distribution_stats(auc_data):
    get_same_cancer_type = lambda x,y: get_sample_cancer_type(x)==get_sample_cancer_type(y)
    same_sample_decrease_store = {True:[],False:[]}
    for sample_id,sample_data in auc_data.items():
        if sample_id =='244':
            continue
        
        for out_sample_id,out_auc in sample_data.items():
            if out_sample_id ==sample_id or out_sample_id =='244':
                continue
            same_cancer_type = get_same_cancer_type(sample_id,out_sample_id)
            print(sample_id,out_sample_id,same_cancer_type,sample_data[sample_id]-out_auc)
            same_sample_decrease_store[same_cancer_type].append(sample_data[sample_id]-out_auc)
        
        
    
    return {same_sample:np.mean(same_sample_decrease_store[same_sample]) for same_sample in (True,False)}
        
def write_different_sample_distribution_text(auc_data):

    #, with a mean decrease in AUC of 0.150 .... the average reduction in test set AUC was only 0.03 between samples of the same type. 
    sample_stats = get_different_sample_distribution_stats(auc_data)
    out_text = f' with a mean decrease in AUC of {sample_stats[False]:.3f} of cancers with the different type compare to {sample_stats[True]:.3f} of the same type'

    out_path = LATEX_TABLE_DIR/'same_different_sample_training.txt'
    with open(out_path,'w') as out:
        out.write(out_text)
    
def plot_different_sample_distribution(sample_data_summary):
    figure_data = sample_data_summary.filter(pl.col('mode')=='Test')
    figure_data = figure_data.filter(~pl.col('add_normal'))    
    for metric in ['AUC','MCC','F1']:
        metric_data =figure_data.to_pandas().set_index(['model_sample_id','out_sample_id'])[metric].unstack(fill_value=0).to_dict()

        if metric =='AUC':
            write_different_sample_distribution_text(metric_data)
        
        out_path = SUPPLEMENTARY_FIGURE_DIR/f'different_sample_distribution_{metric}.png'
        plot_grouped_bar_chart(metric_data,out_path,y_label=f'Test {metric}',legend_title='Training Sample',x_label='Testing Sample',figsize=(22,4),color_scheme=plotting_tools.sample_color_scheme)

def get_calibration_data(probability_data):
    probability_data = probability_data.filter(pl.col('Chromosome').is_in(['chr4','chr5','ch21','chr22']))
    probability_data = probability_data.filter(pl.col('out_sample_id')==pl.col('model_sample_id'))
    bins = np.linspace(0,1,21)
    calibration_data = []
    print(probability_data)

    for sample_id,sample_data in probability_data.group_by('out_sample_id'):
        for bin_index in range(bins.size-1):
            bin_data = sample_data.filter(pl.col('probability').is_between(bins[bin_index],bins[bin_index+1]))
            true_frac = bin_data['tumor_read'].mean()

            bin_entry = {'sample_id':sample_id,'bin_low':bins[bin_index],'bin_high':bins[bin_index+1],'observed_fraction':true_frac,'n_observations':len(bin_data)}
            calibration_data.append(bin_entry)
    calibration_data = pl.DataFrame(calibration_data)
    calibration_data['Bin_Midpoint'] = (calibration_data['bin_low'] + calibration_data['bin_high'])/2
    return calibration_data
        
def plot_probability_calibrations(sample_data):
    calibration_data = get_calibration_data(sample_data)

    plt_index = 0
    fig,axs =plt.subplots(3,2,figsize=(10,10))
    axs = axs.flatten()
    box_size = 0.05
    
    for sample_id,sample_data in calibration_data.group_by('sample_id'):
        #axs[plt_index].scatter(sample_data['Bin_Midpoint'],sample_data['observed_fraction'])
        #axs[plt_index].plot([0,1],[0,1],linestyle='dashed',color='red')
        for index,bin_row in sample_data.iter_rows(named=True):
            rect = Rectangle((bin_row['bin_low'], bin_row['bin_low']), 
                    box_size, box_size)
            axs[plt_index].add_patch(rect)
            axs[plt_index].plot([bin_row['bin_low'], bin_row['bin_high']],[bin_row['observed_fraction'],bin_row['observed_fraction']],color='red')
        
        
        axs[plt_index].set_xlabel('Measured Probability')
        axs[plt_index].set_ylabel('True Probability')
        axs[plt_index].set_title(sample_id)
        axs[plt_index].spines['top'].set_visible(False)
        axs[plt_index].spines['right'].set_visible(False)
        plt_index+=1
    plt.tight_layout()
    plt.savefig(SUPPLEMENTARY_FIGURE_DIR/'test_eval_probability_calibration.png')
    plt.savefig(SUPPLEMENTARY_FIGURE_DIR/'test_eval_probability_calibration.pdf')
    plt.close(fig)


    plt_index = 0
    fig,axs =plt.subplots(3,2,figsize=(10,10))
    axs = axs.flatten()
    for sample_id,sample_data in calibration_data.group_by('sample_id'):
        axs[plt_index].bar(sample_data['Bin_Midpoint'],sample_data['n_observations'],align='center',width=0.04)
        
        axs[plt_index].set_xlabel('Measured Probability')
        axs[plt_index].set_ylabel('Number of reads')
        axs[plt_index].set_title(sample_id)
        axs[plt_index].spines['top'].set_visible(False)
        axs[plt_index].spines['right'].set_visible(False)
        plt_index+=1
    plt.tight_layout()
    plt.savefig(SUPPLEMENTARY_FIGURE_DIR/'test_eval_probability_distribution.png')
    plt.savefig(SUPPLEMENTARY_FIGURE_DIR/'test_eval_probability_distribution.pdf')
    plt.close(fig)

def load_predicted_probabilities(sample_id):
    in_dir = BASE_PREDICTION_DIR/f'{sample_id}_add_normal_False/full_datasets/'
    filename = f'train_{sample_id}_add_normal_False_out_{sample_id}_all_reads.parquet'
    in_path = in_dir/filename
    in_df =pl.read_parquet(in_path)
    #in_df = in_df.rename({'Sample_ID':'sample_id','Read_Index':'read_index','Chromosome':'chromosome','Tumor_Probability':'tumor_probability'})
    
    return in_df['tumor_probability'].to_numpy()
def plot_overall_probability_distributions(sample_ids):
    
    bins = np.linspace(0,1,21)
    bin_width = bins[1]-bins[0]
    bin_midpoints = (bins[:-1]+bins[1:])/2
    fig,axs =plt.subplots(3,2,figsize=(10,10))
    axs = axs.flatten()
    for i,sample_id in enumerate(sample_ids):
        probabilities = load_predicted_probabilities(sample_id)
        probability_hist = np.histogram(probabilities,bins=bins)[0]
        axs[i].bar(bin_midpoints,probability_hist,align='center',width=bin_width*0.8)
        axs[i].set_xlabel('Measured Probability')
        axs[i].set_ylabel('Number of reads')
        axs[i].set_title(plotting_tools.get_sample_mapping()[sample_id.split('_')[0]])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(SUPPLEMENTARY_FIGURE_DIR/'overall_probability_distribution.png')
    plt.savefig(SUPPLEMENTARY_FIGURE_DIR/'overall_probability_distribution.pdf')
    

def get_different_sample_aggregation(sample_data,metric):

    out_type_enum = pl.Enum(['Ovarian','Prostate'])
    
    sample_data = sample_data.with_columns(
        pl.col('out_sample_id').map_elements(get_sample_cancer_type).alias('out_type').cast(out_type_enum),
        pl.col('model_sample_id').map_elements(get_sample_cancer_type).alias('model_type')
    )
    
    cancer_comparison_enum =pl.Enum(['Same Sample','Same Cancer Type','Different Cancer Type'])
    sample_data = sample_data.with_columns(
        pl.when(pl.col("out_sample_id") == pl.col("model_sample_id"))
        .then(pl.lit("Same Sample"))
        .when(pl.col("out_type") == pl.col("model_type"))
        .then(pl.lit("Same Cancer Type"))
        .otherwise(pl.lit("Different Cancer Type"))
        .alias("cancer_compare").cast(cancer_comparison_enum)
    )
    sample_data = sample_data.group_by(['out_type', 'cancer_compare']).agg([
    pl.col(metric).mean().alias('mean'),
    pl.col(metric).std().alias('std'),
    pl.col(metric).count().alias('size')
    ])

    return sample_data.sort(['cancer_compare','out_type'])
    
def plot_different_sample_aggregation(sample_data):
    sample_data = sample_data.filter((pl.col('out_sample_id')!='244') & (pl.col('out_sample_id')!='244'))
    sample_data = sample_data.filter((pl.col('mode')=='Test') & (pl.col('add_normal')==False))
    
    
    
    color_mapping = {'Same Sample':'#5688C7','Same Cancer Type':'#5FAD56','Different Cancer Type':'#B4436C'}
    for metric in ['AUC','MCC','F1']:
        different_sample_aggregation = get_different_sample_aggregation(sample_data,metric)
        
        fig,ax = plt.subplots(1,1,figsize=(6,3))
        bar_width = 0.25
        offset = -bar_width*1.5
        for cancer_compare,compare_table in different_sample_aggregation.group_by('cancer_compare',maintain_order=True):
            cancer_compare = cancer_compare[0]
            
            bars = ax.bar(np.arange(len(compare_table))+offset,compare_table['mean'],yerr=compare_table['std'],label=cancer_compare,capsize=5,align='edge',width=bar_width,color=color_mapping[cancer_compare])

            for i,bar in enumerate(bars):
                height = compare_table['mean'][i]+compare_table['std'][i]+0.005
                
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{compare_table["mean"][i]:.3f}', ha='center', va='bottom', fontsize=8)
            
            offset += bar_width
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',title='Training Data')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Ovarian','Prostate'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(f'Test {metric}')
        plt.tight_layout()
        
        if metric =='AUC':
            plt.savefig(MAIN_FIGURE_DIR/f'different_sample_aggregated_{metric}.pdf')
            plt.savefig(MAIN_FIGURE_DIR/f'different_sample_aggregated_{metric}.png')
        else:

            plt.savefig(SUPPLEMENTARY_FIGURE_DIR/f'different_sample_aggregated_{metric}.png')
            plt.savefig(SUPPLEMENTARY_FIGURE_DIR/f'different_sample_aggregated_{metric}.pdf')
       
        

def get_n_epochs(sample_ids):
    epoch_data = []
    log_dir = Path('/hot/user/tobybaker/ROCIT_Paper/models/main_models' )
    for sample_id in sample_ids:
        in_dir = log_dir /f'{sample_id}_add_normal_False/version_0/' 
        in_path = in_dir/'metrics.csv'
        
        in_df = pl.read_csv(in_path)
        
        in_df = in_df.drop_nulls(subset=['val_loss'])
        
        n_epochs = in_df.filter(pl.col("val_loss") == pl.col("val_loss").min()).select("epoch").item()+1
        
        epoch_data.append(n_epochs)
    return epoch_data


def write_epoch_stats(sample_ids):
    epoch_data= get_n_epochs(sample_ids)
    epoch_median = np.median(epoch_data)
    epoch_range = np.min(epoch_data),np.max(epoch_data)
    out_string = f'Using early stopping on our validation set (Methods), the finished training after a median of {epoch_median} epochs, with a range of {epoch_range[0]}-{epoch_range[1]} epochs across the {len(epoch_data)} samples in our cohort.'
    with open(LATEX_TABLE_DIR/'epoch_training_stats.txt','w') as out:
        out.write(out_string)


if __name__ =='__main__':
    sample_ids = ['216_TU','244_TU','264_TU','053_TU','BS14772_TU','BS15145_TU']
    plot_custom_training_runs.plot_custom_training_runs()
    
    write_epoch_stats(sample_ids)

    
    
    sample_data,sample_data_summary = load_sample_data(sample_ids)
    
    plot_different_sample_aggregation(sample_data_summary)
    
    plot_different_sample_distribution(sample_data_summary)
    
    
    plot_main_figure(sample_data_summary)

    
    #plot_probability_calibrations(sample_data)
    #plot_different_sample_heatmap(sample_data_summary)
    
    
    

    plot_overall_probability_distributions(sample_ids)

    
    latex_page_maker.get_latex_document(sample_data,sample_data_summary)
    

    
    plot_dataset_distributions(sample_data_summary)
    plot_add_normal_distributions(sample_data_summary)
    
    
    

    
    